from __future__ import absolute_import, division, print_function
import traceback
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_deploy_azure_vm(self):
    """
        Get Cloud Manager connector for AZURE
        :return:
            Dictionary of current details if Cloud Manager connector for AZURE
            None if Cloud Manager connector for AZURE is not found
        """
    exists = False
    resource_client = get_client_from_cli_profile(ResourceManagementClient)
    try:
        exists = resource_client.deployments.check_existence(self.parameters['resource_group'], self.parameters['name'])
    except CloudError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    if not exists:
        return None
    return exists