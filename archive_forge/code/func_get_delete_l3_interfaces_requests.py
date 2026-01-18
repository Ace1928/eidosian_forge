from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_l3_interfaces_requests(self, want, have):
    requests = []
    ipv4_addrs_url_all = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4/addresses'
    ipv6_addrs_url_all = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/addresses'
    ipv4_anycast_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4'
    ipv4_anycast_url += '/openconfig-interfaces-ext:sag-ipv4/config/static-anycast-gateway={anycast_ip}'
    ipv4_addr_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv4/addresses/address={address}'
    ipv6_addr_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/addresses/address={address}'
    ipv6_enabled_url = 'data/openconfig-interfaces:interfaces/interface={intf_name}/{sub_intf_name}/openconfig-if-ip:ipv6/config/enabled'
    if not want:
        return requests
    for each_l3 in want:
        l3 = each_l3.copy()
        name = l3.pop('name')
        sub_intf = self.get_sub_interface_name(name)
        have_obj = next((e_cfg for e_cfg in have if e_cfg['name'] == name), None)
        if not have_obj:
            continue
        have_ipv4_addrs = list()
        have_ipv4_anycast_addrs = list()
        have_ipv6_addrs = list()
        have_ipv6_enabled = None
        if have_obj.get('ipv4'):
            if 'addresses' in have_obj['ipv4']:
                have_ipv4_addrs = have_obj['ipv4']['addresses']
            if 'anycast_addresses' in have_obj['ipv4']:
                have_ipv4_anycast_addrs = have_obj['ipv4']['anycast_addresses']
        have_ipv6_addrs = self.get_address('ipv6', [have_obj])
        if have_obj.get('ipv6') and 'enabled' in have_obj['ipv6']:
            have_ipv6_enabled = have_obj['ipv6']['enabled']
        ipv4 = l3.get('ipv4', None)
        ipv6 = l3.get('ipv6', None)
        ipv4_addrs = None
        ipv6_addrs = None
        is_del_ipv4 = None
        is_del_ipv6 = None
        if name and ipv4 is None and (ipv6 is None):
            is_del_ipv4 = True
            is_del_ipv6 = True
        elif ipv4 and (not ipv4.get('addresses')) and (not ipv4.get('anycast_addresses')):
            is_del_ipv4 = True
        elif ipv6 and (not ipv6.get('addresses')) and (ipv6.get('enabled') is None):
            is_del_ipv6 = True
        if is_del_ipv4:
            if have_ipv4_addrs and len(have_ipv4_addrs) != 0:
                ipv4_addrs_delete_request = {'path': ipv4_addrs_url_all.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
                requests.append(ipv4_addrs_delete_request)
            if have_ipv4_anycast_addrs and len(have_ipv4_anycast_addrs) != 0:
                for ip in have_ipv4_anycast_addrs:
                    ip = ip.replace('/', '%2f')
                    anycast_delete_request = {'path': ipv4_anycast_url.format(intf_name=name, sub_intf_name=sub_intf, anycast_ip=ip), 'method': DELETE}
                    requests.append(anycast_delete_request)
        else:
            ipv4_addrs = []
            ipv4_anycast_addrs = []
            if l3.get('ipv4'):
                if l3['ipv4'].get('addresses'):
                    ipv4_addrs = l3['ipv4']['addresses']
                if l3['ipv4'].get('anycast_addresses'):
                    ipv4_anycast_addrs = l3['ipv4']['anycast_addresses']
            ipv4_del_reqs = []
            if ipv4_addrs:
                for ip in ipv4_addrs:
                    if have_ipv4_addrs:
                        match_ip = next((addr for addr in have_ipv4_addrs if addr['address'] == ip['address']), None)
                        if match_ip:
                            addr = ip['address'].split('/')[0]
                            del_url = ipv4_addr_url.format(intf_name=name, sub_intf_name=sub_intf, address=addr)
                            if match_ip['secondary']:
                                del_url += '/config/secondary'
                                ipv4_del_reqs.insert(0, {'path': del_url, 'method': DELETE})
                            else:
                                ipv4_del_reqs.append({'path': del_url, 'method': DELETE})
                        if ipv4_del_reqs:
                            requests.extend(ipv4_del_reqs)
            if ipv4_anycast_addrs:
                for ip in ipv4_anycast_addrs:
                    if have_ipv4_anycast_addrs and ip in have_ipv4_anycast_addrs:
                        ip = ip.replace('/', '%2f')
                        anycast_delete_request = {'path': ipv4_anycast_url.format(intf_name=name, sub_intf_name=sub_intf, anycast_ip=ip), 'method': DELETE}
                        requests.append(anycast_delete_request)
        if is_del_ipv6:
            if have_ipv6_addrs and len(have_ipv6_addrs) != 0:
                ipv6_addrs_delete_request = {'path': ipv6_addrs_url_all.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
                requests.append(ipv6_addrs_delete_request)
            if have_ipv6_enabled:
                ipv6_enabled_delete_request = {'path': ipv6_enabled_url.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
                requests.append(ipv6_enabled_delete_request)
        else:
            ipv6_addrs = []
            ipv6_enabled = None
            if l3.get('ipv6'):
                if l3['ipv6'].get('addresses'):
                    ipv6_addrs = l3['ipv6']['addresses']
                if 'enabled' in l3['ipv6']:
                    ipv6_enabled = l3['ipv6']['enabled']
            if ipv6_addrs:
                for ip in ipv6_addrs:
                    if have_ipv6_addrs and ip['address'] in have_ipv6_addrs:
                        addr = ip['address'].split('/')[0]
                        request = {'path': ipv6_addr_url.format(intf_name=name, sub_intf_name=sub_intf, address=addr), 'method': DELETE}
                        requests.append(request)
            if have_ipv6_enabled and ipv6_enabled is not None:
                request = {'path': ipv6_enabled_url.format(intf_name=name, sub_intf_name=sub_intf), 'method': DELETE}
                requests.append(request)
    return requests