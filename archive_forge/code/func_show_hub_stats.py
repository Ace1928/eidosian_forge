from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def show_hub_stats(self, resource_group, name):
    try:
        return self.IoThub_client.iot_hub_resource.get_stats(resource_group, name).as_dict()
    except Exception as exc:
        self.fail('Failed to getting statistics for IoT Hub {0}/{1}: {2}'.format(resource_group, name, str(exc)))