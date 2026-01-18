from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_modify_mac_requests(self, commands):
    requests = []
    if not commands:
        return requests
    for cmd in commands:
        vrf_name = cmd.get('vrf_name', None)
        mac = cmd.get('mac', {})
        if mac:
            aging_time = mac.get('aging_time', None)
            dampening_interval = mac.get('dampening_interval', None)
            dampening_threshold = mac.get('dampening_threshold', None)
            mac_table_entries = mac.get('mac_table_entries', [])
            fdb_dict = {}
            dampening_cfg_dict = {}
            if aging_time:
                fdb_dict['config'] = {'mac-aging-time': aging_time}
            if dampening_interval:
                dampening_cfg_dict['interval'] = dampening_interval
            if dampening_threshold:
                dampening_cfg_dict['threshold'] = dampening_threshold
            if mac_table_entries:
                entry_list = []
                entries_dict = {}
                mac_table_dict = {}
                for entry in mac_table_entries:
                    entry_dict = {}
                    entry_cfg_dict = {}
                    mac_address = entry.get('mac_address', None)
                    vlan_id = entry.get('vlan_id', None)
                    interface = entry.get('interface', None)
                    if mac_address:
                        entry_dict['mac-address'] = mac_address
                        entry_cfg_dict['mac-address'] = mac_address
                    if vlan_id:
                        entry_dict['vlan'] = vlan_id
                        entry_cfg_dict['vlan'] = vlan_id
                    if entry_cfg_dict:
                        entry_dict['config'] = entry_cfg_dict
                    if interface:
                        entry_dict['interface'] = {'interface-ref': {'config': {'interface': interface, 'subinterface': 0}}}
                    if entry_dict:
                        entry_list.append(entry_dict)
                if entry_list:
                    entries_dict['entry'] = entry_list
                if entries_dict:
                    mac_table_dict['entries'] = entries_dict
                if mac_table_dict:
                    fdb_dict['mac-table'] = mac_table_dict
            if fdb_dict:
                url = '%s=%s/fdb' % (NETWORK_INSTANCE_PATH, vrf_name)
                payload = {'openconfig-network-instance:fdb': fdb_dict}
                requests.append({'path': url, 'method': PATCH, 'data': payload})
            if dampening_cfg_dict:
                url = '%s=%s/openconfig-mac-dampening:mac-dampening' % (NETWORK_INSTANCE_PATH, vrf_name)
                payload = {'openconfig-mac-dampening:mac-dampening': {'config': dampening_cfg_dict}}
                requests.append({'path': url, 'method': PATCH, 'data': payload})
    return requests