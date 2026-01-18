from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def set_wins_registration(self, mgrattr):
    Key = mgrattr['mgr_attr_name']
    nic_info = self.get_manager_ethernet_uri()
    ethuri = nic_info['nic_addr']
    payload = {'Oem': {'Hpe': {'IPv4': {Key: False}}}}
    response = self.patch_request(self.root_uri + ethuri, payload)
    if not response['ret']:
        return response
    return {'ret': True, 'changed': True, 'msg': 'Modified %s' % mgrattr['mgr_attr_name']}