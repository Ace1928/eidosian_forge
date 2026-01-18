from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l3_interfaces.l3_interfaces import L3_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_l3_interfaces(self):
    url = 'data/openconfig-interfaces:interfaces/interface'
    method = 'GET'
    request = [{'path': url, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    l3_lists = []
    if 'openconfig-interfaces:interface' in response[0][1]:
        l3_lists = response[0][1].get('openconfig-interfaces:interface', [])
    l3_configs = []
    for l3 in l3_lists:
        l3_dict = dict()
        l3_name = l3['name']
        if l3_name == 'eth0':
            continue
        l3_dict['name'] = l3_name
        ip = None
        anycast_addr = list()
        if l3.get('openconfig-vlan:routed-vlan'):
            ip = l3['openconfig-vlan:routed-vlan']
            if ip.get('openconfig-if-ip:ipv4', None) and ip['openconfig-if-ip:ipv4'].get('openconfig-interfaces-ext:sag-ipv4', None):
                if ip['openconfig-if-ip:ipv4']['openconfig-interfaces-ext:sag-ipv4'].get('config', None):
                    if ip['openconfig-if-ip:ipv4']['openconfig-interfaces-ext:sag-ipv4']['config'].get('static-anycast-gateway', None):
                        anycast_addr = ip['openconfig-if-ip:ipv4']['openconfig-interfaces-ext:sag-ipv4']['config']['static-anycast-gateway']
        else:
            ip = l3.get('subinterfaces', {}).get('subinterface', [{}])[0]
        l3_dict['ipv4'] = dict()
        l3_ipv4 = list()
        if anycast_addr:
            l3_dict['ipv4']['anycast_addresses'] = anycast_addr
        elif 'openconfig-if-ip:ipv4' in ip and 'addresses' in ip['openconfig-if-ip:ipv4'] and ('address' in ip['openconfig-if-ip:ipv4']['addresses']):
            for ipv4 in ip['openconfig-if-ip:ipv4']['addresses']['address']:
                if ipv4.get('config') and ipv4.get('config').get('ip'):
                    temp = dict()
                    temp['address'] = str(ipv4['config']['ip']) + '/' + str(ipv4['config']['prefix-length'])
                    temp['secondary'] = ipv4['config']['secondary']
                    l3_ipv4.append(temp)
            if l3_ipv4:
                l3_dict['ipv4']['addresses'] = l3_ipv4
        l3_dict['ipv6'] = dict()
        l3_ipv6 = list()
        if 'openconfig-if-ip:ipv6' in ip:
            if 'addresses' in ip['openconfig-if-ip:ipv6'] and 'address' in ip['openconfig-if-ip:ipv6']['addresses']:
                for ipv6 in ip['openconfig-if-ip:ipv6']['addresses']['address']:
                    if ipv6.get('config') and ipv6.get('config').get('ip'):
                        temp = dict()
                        temp['address'] = str(ipv6['config']['ip']) + '/' + str(ipv6['config']['prefix-length'])
                        l3_ipv6.append(temp)
                if l3_ipv6:
                    l3_dict['ipv6']['addresses'] = l3_ipv6
            if 'config' in ip['openconfig-if-ip:ipv6'] and 'enabled' in ip['openconfig-if-ip:ipv6']['config']:
                l3_dict['ipv6']['enabled'] = ip['openconfig-if-ip:ipv6']['config']['enabled']
        l3_configs.append(l3_dict)
    return l3_configs