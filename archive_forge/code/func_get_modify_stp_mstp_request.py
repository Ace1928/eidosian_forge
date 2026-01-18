from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_stp_mstp_request(self, commands, have):
    request = None
    if not commands:
        return request
    mstp = commands.get('mstp', None)
    if mstp:
        mstp_dict = {}
        config_dict = {}
        mst_name = mstp.get('mst_name', None)
        revision = mstp.get('revision', None)
        max_hop = mstp.get('max_hop', None)
        hello_time = mstp.get('hello_time', None)
        max_age = mstp.get('max_age', None)
        fwd_delay = mstp.get('fwd_delay', None)
        mst_instances = mstp.get('mst_instances', None)
        if mst_name:
            config_dict['name'] = mst_name
        if revision:
            config_dict['revision'] = revision
        if max_hop:
            config_dict['max-hop'] = max_hop
        if hello_time:
            config_dict['hello-time'] = hello_time
        if max_age:
            config_dict['max-age'] = max_age
        if fwd_delay:
            config_dict['forwarding-delay'] = fwd_delay
        if mst_instances:
            mst_inst_list = []
            pop_list = []
            for mst in mst_instances:
                mst_inst_dict = {}
                mst_cfg_dict = {}
                mst_index = mst_instances.index(mst)
                mst_id = mst.get('mst_id', None)
                bridge_priority = mst.get('bridge_priority', None)
                interfaces = mst.get('interfaces', None)
                vlans = mst.get('vlans', None)
                if mst_id:
                    mst_cfg_dict['mst-id'] = mst_id
                if bridge_priority:
                    mst_cfg_dict['bridge-priority'] = bridge_priority
                if interfaces:
                    intf_list = self.get_interfaces_list(interfaces)
                    if intf_list:
                        mst_inst_dict['interfaces'] = {'interface': intf_list}
                if vlans:
                    if have:
                        cfg_mstp = have.get('mstp', None)
                        if cfg_mstp:
                            cfg_mst_instances = cfg_mstp.get('mst_instances', None)
                            if cfg_mst_instances:
                                for cfg_mst in cfg_mst_instances:
                                    cfg_mst_id = cfg_mst.get('mst_id', None)
                                    cfg_vlans = cfg_mst.get('vlans', None)
                                    if mst_id == cfg_mst_id and cfg_vlans:
                                        vlans = self.get_vlans_diff(vlans, cfg_vlans)
                                        if not vlans:
                                            pop_list.insert(0, mst_index)
                    if vlans:
                        mst_cfg_dict['vlan'] = self.convert_vlans_list(vlans)
                if mst_cfg_dict:
                    mst_inst_dict['mst-id'] = mst_id
                    mst_inst_dict['config'] = mst_cfg_dict
                if mst_inst_dict:
                    mst_inst_list.append(mst_inst_dict)
            if pop_list:
                for i in pop_list:
                    commands['mstp']['mst_instances'][i].pop('vlans')
            if mst_inst_list:
                mstp_dict['mst-instances'] = {'mst-instance': mst_inst_list}
        if config_dict:
            mstp_dict['config'] = config_dict
        if mstp_dict:
            url = '%s/mstp' % STP_PATH
            payload = {'openconfig-spanning-tree:mstp': mstp_dict}
            request = {'path': url, 'method': PATCH, 'data': payload}
    return request