from __future__ import (absolute_import, division, print_function)
import sys
@property
def netapp_client(self):
    self.log('Getting netapp client')
    if self._new_style is None:
        self.fail_when_import_errors(IMPORT_ERRORS)
    if self._netapp_client is None:
        if self._new_style:
            self._netapp_client = self.get_mgmt_svc_client(NetAppManagementClient)
        else:
            self._netapp_client = self.get_mgmt_svc_client(AzureNetAppFilesManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2018-05-01')
    return self._netapp_client