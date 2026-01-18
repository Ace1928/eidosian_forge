from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
def raw_post_resource(self, resource_uri, request_body):
    if resource_uri is None:
        return {'ret': False, 'msg': 'resource_uri is missing'}
    resource_uri_has_actions = True
    if '/Actions/' not in resource_uri:
        resource_uri_has_actions = False
    if request_body is None:
        return {'ret': False, 'msg': 'request_body is missing'}
    action_base_uri = resource_uri.split('/Actions/')[0]
    response = self.get_request(self.root_uri + action_base_uri)
    if response['ret'] is False:
        return response
    if 'Actions' not in response['data']:
        if resource_uri_has_actions:
            return {'ret': False, 'msg': 'Actions property not found in %s' % action_base_uri}
        else:
            response['data']['Actions'] = {}
    action_found = False
    action_info_uri = None
    action_target_uri_list = []
    for key in response['data']['Actions'].keys():
        if action_found:
            break
        if not key.startswith('#'):
            continue
        if 'target' in response['data']['Actions'][key]:
            if resource_uri == response['data']['Actions'][key]['target']:
                action_found = True
                if '@Redfish.ActionInfo' in response['data']['Actions'][key]:
                    action_info_uri = response['data']['Actions'][key]['@Redfish.ActionInfo']
            else:
                action_target_uri_list.append(response['data']['Actions'][key]['target'])
    if not action_found and 'Oem' in response['data']['Actions']:
        for key in response['data']['Actions']['Oem'].keys():
            if action_found:
                break
            if not key.startswith('#'):
                continue
            if 'target' in response['data']['Actions']['Oem'][key]:
                if resource_uri == response['data']['Actions']['Oem'][key]['target']:
                    action_found = True
                    if '@Redfish.ActionInfo' in response['data']['Actions']['Oem'][key]:
                        action_info_uri = response['data']['Actions']['Oem'][key]['@Redfish.ActionInfo']
                else:
                    action_target_uri_list.append(response['data']['Actions']['Oem'][key]['target'])
    if not action_found and resource_uri_has_actions:
        return {'ret': False, 'msg': 'Specified resource_uri is not a supported action target uri, please specify a supported target uri instead. Supported uri: %s' % str(action_target_uri_list)}
    if action_info_uri is not None:
        response = self.get_request(self.root_uri + action_info_uri)
        if response['ret'] is False:
            return response
        for key in request_body.keys():
            key_found = False
            for para in response['data']['Parameters']:
                if key == para['Name']:
                    key_found = True
                    break
            if not key_found:
                return {'ret': False, 'msg': 'Invalid property %s found in request_body. Please refer to @Redfish.ActionInfo Parameters: %s' % (key, str(response['data']['Parameters']))}
    response = self.post_request(self.root_uri + resource_uri, request_body)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True}