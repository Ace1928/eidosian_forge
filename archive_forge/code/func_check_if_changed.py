from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def check_if_changed(self, parameter_name, old_response):
    """"
        Compute if there is an update to the resource or not

        :return: True if resource is changed compared to the current one
        """
    if parameter_name in self.parameters and (parameter_name not in old_response or self.parameters[parameter_name] != old_response[parameter_name]):
        return True
    elif parameter_name not in self.parameters and parameter_name in old_response:
        return True
    else:
        return False