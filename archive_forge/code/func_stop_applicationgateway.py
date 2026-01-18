from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def stop_applicationgateway(self):
    self.log('Stopping the Application Gateway instance {0}'.format(self.name))
    try:
        response = self.network_client.application_gateways.begin_stop(resource_group_name=self.resource_group, application_gateway_name=self.name)
        if isinstance(response, LROPoller):
            self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to stop the Application Gateway instance.')
        self.fail('Error stopping the Application Gateway instance: {0}'.format(str(e)))