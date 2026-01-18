from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_snapmirror(self):
    source_we_info, dest_we_info, err = self.na_helper.get_working_environment_detail_for_snapmirror(self.rest_api, self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg=err)
    get_url = '/occm/api/replication/status/%s' % source_we_info['publicId']
    snapmirror_info, err, dummy = self.rest_api.send_request('GET', get_url, None, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error getting snapmirror relationship %s: %s.' % (err, snapmirror_info))
    sm_found = False
    snapmirror = None
    for sm in snapmirror_info:
        if sm['destination']['volumeName'] == self.parameters['destination_volume_name']:
            sm_found = True
            snapmirror = sm
            break
    if not sm_found:
        return None
    result = {'source_working_environment_id': source_we_info['publicId'], 'destination_svm_name': snapmirror['destination']['svmName'], 'destination_working_environment_id': dest_we_info['publicId']}
    if not dest_we_info['publicId'].startswith('fs-'):
        result['cloud_provider_name'] = dest_we_info['cloudProviderName']
    return result