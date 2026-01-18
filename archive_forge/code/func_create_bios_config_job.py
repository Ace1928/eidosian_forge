from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.common.text.converters import to_native
def create_bios_config_job(self):
    result = {}
    key = 'Bios'
    jobs = 'Jobs'
    response = self.get_request(self.root_uri + self.systems_uris[0])
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if key not in data:
        return {'ret': False, 'msg': 'Key %s not found' % key}
    bios_uri = data[key]['@odata.id']
    response = self.get_request(self.root_uri + bios_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    set_bios_attr_uri = data['@Redfish.Settings']['SettingsObject']['@odata.id']
    payload = {'TargetSettingsURI': set_bios_attr_uri}
    response = self.post_request(self.root_uri + self.manager_uri + '/' + jobs, payload)
    if response['ret'] is False:
        return response
    response_output = response['resp'].__dict__
    job_id_full = response_output['headers']['Location']
    job_id = re.search('JID_.+', job_id_full).group()
    return {'ret': True, 'msg': 'Config job %s created' % job_id, 'job_id': job_id_full}