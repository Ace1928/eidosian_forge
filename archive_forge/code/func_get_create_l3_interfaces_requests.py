from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_create_l3_interfaces_requests(self, configs, have, want):
    requests = []
    if not configs:
        return requests
    ipv4_addrs_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4/addresses'
    ipv4_anycast_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4/'
    ipv4_anycast_url += 'openconfig-interfaces-ext:sag-ipv4/config/static-anycast-gateway'
    ipv6_addrs_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/addresses'
    ipv6_enabled_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/config'
    for l3 in configs:
        l3_interface_name = l3.get('name')
        if l3_interface_name == 'eth0':
            continue
        sub_intf = self.get_sub_interface_name(l3_interface_name)
        ipv4_addrs = []
        ipv4_anycast = []
        if l3.get('ipv4'):
            if l3['ipv4'].get('addresses'):
                ipv4_addrs = l3['ipv4']['addresses']
            if l3['ipv4'].get('anycast_addresses'):
                ipv4_anycast = l3['ipv4']['anycast_addresses']
        ipv6_addrs = []
        ipv6_enabled = None
        if l3.get('ipv6'):
            if l3['ipv6'].get('addresses'):
                ipv6_addrs = l3['ipv6']['addresses']
            if 'enabled' in l3['ipv6']:
                ipv6_enabled = l3['ipv6']['enabled']
        if ipv4_addrs:
            ipv4_addrs_pri_payload = []
            ipv4_addrs_sec_payload = []
            for item in ipv4_addrs:
                ipv4_addr_mask = item['address'].split('/')
                ipv4 = ipv4_addr_mask[0]
                ipv4_mask = ipv4_addr_mask[1]
                ipv4_secondary = item['secondary']
                if ipv4_secondary:
                    ipv4_addrs_sec_payload.append(self.build_create_addr_payload(ipv4, ipv4_mask, ipv4_secondary))
                else:
                    ipv4_addrs_pri_payload.append(self.build_create_addr_payload(ipv4, ipv4_mask, ipv4_secondary))
            if ipv4_addrs_pri_payload:
                payload = self.build_create_payload(ipv4_addrs_pri_payload)
                ipv4_addrs_req = {'path': ipv4_addrs_url.format(intf_name=l3_interface_name, sub_intf_name=sub_intf), 'method': PATCH, 'data': payload}
                requests.append(ipv4_addrs_req)
            if ipv4_addrs_sec_payload:
                payload = self.build_create_payload(ipv4_addrs_sec_payload)
                ipv4_addrs_req = {'path': ipv4_addrs_url.format(intf_name=l3_interface_name, sub_intf_name=sub_intf), 'method': PATCH, 'data': payload}
                requests.append(ipv4_addrs_req)
        if ipv4_anycast:
            anycast_payload = {'openconfig-interfaces-ext:static-anycast-gateway': ipv4_anycast}
            anycast_url = ipv4_anycast_url.format(intf_name=l3_interface_name, sub_intf_name=sub_intf)
            requests.append({'path': anycast_url, 'method': PATCH, 'data': anycast_payload})
        if ipv6_addrs:
            ipv6_addrs_payload = []
            for item in ipv6_addrs:
                ipv6_addr_mask = item['address'].split('/')
                ipv6 = ipv6_addr_mask[0]
                ipv6_mask = ipv6_addr_mask[1]
                ipv6_addrs_payload.append(self.build_create_addr_payload(ipv6, ipv6_mask))
            if ipv6_addrs_payload:
                payload = self.build_create_payload(ipv6_addrs_payload)
                ipv6_addrs_req = {'path': ipv6_addrs_url.format(intf_name=l3_interface_name, sub_intf_name=sub_intf), 'method': PATCH, 'data': payload}
                requests.append(ipv6_addrs_req)
        if ipv6_enabled is not None:
            payload = self.build_update_ipv6_enabled(ipv6_enabled)
            ipv6_enabled_req = {'path': ipv6_enabled_url.format(intf_name=l3_interface_name, sub_intf_name=sub_intf), 'method': PATCH, 'data': payload}
            requests.append(ipv6_enabled_req)
    return requests