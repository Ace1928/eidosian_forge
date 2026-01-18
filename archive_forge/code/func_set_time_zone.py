from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def set_time_zone(self, attr):
    key = attr['mgr_attr_name']
    uri = self.manager_uri + 'DateTime/'
    response = self.get_request(self.root_uri + uri)
    if not response['ret']:
        return response
    data = response['data']
    if key not in data:
        return {'ret': False, 'changed': False, 'msg': 'Key %s not found' % key}
    timezones = data['TimeZoneList']
    index = ''
    for tz in timezones:
        if attr['mgr_attr_value'] in tz['Name']:
            index = tz['Index']
            break
    payload = {key: {'Index': index}}
    response = self.patch_request(self.root_uri + uri, payload)
    if not response['ret']:
        return response
    return {'ret': True, 'changed': True, 'msg': 'Modified %s' % attr['mgr_attr_name']}