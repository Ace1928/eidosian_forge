from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.vlans.vlans import (
def parse_vlan(self, data):
    objs = []
    mtu_objs = []
    remote_objs = []
    final_objs = []
    pvlan_objs = []
    config = data.split('\n')
    vlan_info = ''
    temp = ''
    vlan_name = True
    for conf in config:
        if len(list(filter(None, conf.split(' ')))) <= 2 and vlan_name:
            temp = temp + conf
            if len(list(filter(None, temp.split(' ')))) <= 2:
                continue
        if 'VLAN Name' in conf:
            vlan_info = 'Name'
        elif 'VLAN Type' in conf:
            vlan_info = 'Type'
            vlan_name = False
        elif 'Remote SPAN' in conf:
            vlan_info = 'Remote'
            vlan_name = False
        elif 'VLAN AREHops' in conf or 'STEHops' in conf:
            vlan_info = 'Hops'
            vlan_name = False
        elif 'Primary Secondary' in conf:
            vlan_info = 'Private'
            vlan_name = False
        if temp:
            conf = temp
            temp = ''
        if conf and ' ' not in filter(None, conf.split('-')) and (not conf.split(' ')[0] == ''):
            obj = self.render_config(self.generated_spec, conf, vlan_info)
            if 'mtu' in obj:
                mtu_objs.append(obj)
            elif 'remote_span' in obj:
                remote_objs = obj
            elif 'tmp_pvlans' in obj:
                pvlan_objs.append(obj)
            elif obj:
                objs.append(obj)
    for o, m in zip(objs, mtu_objs):
        o.update(m)
        final_objs.append(o)
    if remote_objs:
        if remote_objs.get('remote_span'):
            for each in remote_objs.get('remote_span'):
                for every in final_objs:
                    if each == every.get('vlan_id'):
                        every.update({'remote_span': True})
                        break
    if pvlan_objs:
        pvlan_final = {}
        if len(pvlan_objs) > 0:
            for data in pvlan_objs:
                pvdata = data.get('tmp_pvlans')
                privlan = pvdata.get('primary')
                secvlan = pvdata.get('secondary')
                sectype = pvdata.get('sec_type')
                if secvlan and (isinstance(secvlan, int) or secvlan.isnumeric()):
                    secvlan = int(secvlan)
                    pvlan_final[secvlan] = {'private_vlan': {'type': sectype}}
                if privlan and (isinstance(privlan, int) or privlan.isnumeric()):
                    privlan = int(privlan)
                    if privlan not in pvlan_final.keys():
                        pvlan_final[privlan] = {'private_vlan': {'type': 'primary', 'associated': []}}
                    if secvlan and (isinstance(secvlan, int) or secvlan.isnumeric()):
                        pvlan_final[privlan]['private_vlan']['associated'].append(int(secvlan))
            for vlan_id, data in pvlan_final.items():
                for every in final_objs:
                    if vlan_id == every.get('vlan_id'):
                        every.update(data)
    if final_objs:
        return objs
    else:
        return {}