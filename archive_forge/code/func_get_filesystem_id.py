from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def get_filesystem_id(self):
    list_filesystem, error = self.rest_api.get('FileSystems')
    if error:
        self.module.fail_json(msg=error)
    for filesystem in list_filesystem:
        if filesystem['creationToken'] == self.parameters['creationToken']:
            return filesystem['fileSystemId']
    return None