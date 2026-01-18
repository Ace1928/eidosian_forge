from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def set_ntp_server(self, mgr_attributes):
    result = {}
    setkey = mgr_attributes['mgr_attr_name']
    nic_info = self.get_manager_ethernet_uri()
    ethuri = nic_info['nic_addr']
    response = self.get_request(self.root_uri + ethuri)
    if not response['ret']:
        return response
    result['ret'] = True
    data = response['data']
    payload = {'DHCPv4': {'UseNTPServers': ''}}
    if data['DHCPv4']['UseNTPServers']:
        payload['DHCPv4']['UseNTPServers'] = False
        res_dhv4 = self.patch_request(self.root_uri + ethuri, payload)
        if not res_dhv4['ret']:
            return res_dhv4
    payload = {'DHCPv6': {'UseNTPServers': ''}}
    if data['DHCPv6']['UseNTPServers']:
        payload['DHCPv6']['UseNTPServers'] = False
        res_dhv6 = self.patch_request(self.root_uri + ethuri, payload)
        if not res_dhv6['ret']:
            return res_dhv6
    datetime_uri = self.manager_uri + 'DateTime'
    listofips = mgr_attributes['mgr_attr_value'].split(' ')
    if len(listofips) > 2:
        return {'ret': False, 'changed': False, 'msg': 'More than 2 NTP Servers mentioned'}
    ntp_list = []
    for ips in listofips:
        ntp_list.append(ips)
    while len(ntp_list) < 2:
        ntp_list.append('0.0.0.0')
    payload = {setkey: ntp_list}
    response1 = self.patch_request(self.root_uri + datetime_uri, payload)
    if not response1['ret']:
        return response1
    return {'ret': True, 'changed': True, 'msg': 'Modified %s' % mgr_attributes['mgr_attr_name']}