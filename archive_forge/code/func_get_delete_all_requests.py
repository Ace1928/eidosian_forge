from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_all_requests(self, configs):
    requests = []
    ipv4_addrs_url_all = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4/addresses'
    ipv4_anycast_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4'
    ipv4_anycast_url += '/openconfig-interfaces-ext:sag-ipv4/config/static-anycast-gateway={anycast_ip}'
    ipv6_addrs_url_all = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/addresses'
    ipv6_enabled_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/config/enabled'
    for l3 in configs:
        name = l3.get('name')
        ipv4_addrs = []
        ipv4_anycast = []
        if name == 'Management0':
            continue
        if l3.get('ipv4'):
            if l3['ipv4'].get('addresses'):
                ipv4_addrs = l3['ipv4']['addresses']
            if l3['ipv4'].get('anycast_addresses', None):
                ipv4_anycast = l3['ipv4']['anycast_addresses']
        ipv6_addrs = []
        ipv6_enabled = None
        if l3.get('ipv6'):
            if l3['ipv6'].get('addresses'):
                ipv6_addrs = l3['ipv6']['addresses']
            if 'enabled' in l3['ipv6']:
                ipv6_enabled = l3['ipv6']['enabled']
        sub_intf = self.get_sub_interface_name(name)
        if ipv4_addrs:
            ipv4_addrs_delete_request = {'path': ipv4_addrs_url_all.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
            requests.append(ipv4_addrs_delete_request)
        if ipv4_anycast:
            for ip in ipv4_anycast:
                ip = ip.replace('/', '%2f')
                anycast_delete_request = {'path': ipv4_anycast_url.format(intf_name=name, sub_intf_name=sub_intf, anycast_ip=ip), 'method': DELETE}
                requests.append(anycast_delete_request)
        if ipv6_addrs:
            ipv6_addrs_delete_request = {'path': ipv6_addrs_url_all.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
            requests.append(ipv6_addrs_delete_request)
        if ipv6_enabled:
            ipv6_enabled_delete_request = {'path': ipv6_enabled_url.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
            requests.append(ipv6_enabled_delete_request)
    return requests