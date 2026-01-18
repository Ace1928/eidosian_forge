from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
def raw_patch_resource(self, resource_uri, request_body):
    if resource_uri is None:
        return {'ret': False, 'msg': 'resource_uri is missing'}
    if request_body is None:
        return {'ret': False, 'msg': 'request_body is missing'}
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    original_etag = response['data']['@odata.etag']
    data = response['data']
    for key in request_body.keys():
        if key not in data:
            return {'ret': False, 'msg': 'Key %s not found. Supported key list: %s' % (key, str(data.keys()))}
    response = self.patch_request(self.root_uri + resource_uri, request_body)
    if response['ret'] is False:
        return response
    current_etag = ''
    if 'data' in response and '@odata.etag' in response['data']:
        current_etag = response['data']['@odata.etag']
    if current_etag != original_etag:
        return {'ret': True, 'changed': True}
    else:
        return {'ret': True, 'changed': False}