from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_org_container(self):
    params = {'include': 'compliance,region'}
    response, error = self.rest_api.get('api/v3/org/containers', params=params)
    if error:
        self.module.fail_json(msg=error)
    for container in response['data']:
        if container['name'] == self.parameters['name']:
            return container
    return None