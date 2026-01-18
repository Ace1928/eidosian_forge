from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
def raw_get_collection_resource(self, resource_uri):
    if resource_uri is None:
        return {'ret': False, 'msg': 'resource_uri is missing'}
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    if 'Members' not in response['data']:
        return {'ret': False, 'msg': "Specified resource_uri doesn't have Members property"}
    member_list = [i['@odata.id'] for i in response['data'].get('Members', [])]
    data_list = []
    for member_uri in member_list:
        uri = self.root_uri + member_uri
        response = self.get_request(uri)
        if response['ret'] is False:
            return response
        data = response['data']
        data_list.append(data)
    return {'ret': True, 'data_list': data_list}