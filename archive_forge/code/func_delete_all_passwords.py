from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
import datetime
def delete_all_passwords(self, old_passwords):
    if len(old_passwords) == 0:
        self.results['changed'] = False
        return
    try:
        self.client.applications.patch(self.app_object_id, ApplicationUpdateParameters(password_credentials=[]))
        self.results['changed'] = True
    except GraphErrorException as ge:
        self.fail('fail to purge all passwords for app: {0} - {1}'.format(self.app_object_id, str(ge)))