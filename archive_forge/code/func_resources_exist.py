from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def resources_exist(self):
    found = False
    try:
        response = self.rm_client.resources.list_by_resource_group(self.name)
    except AttributeError:
        response = self.rm_client.resource_groups.list_resources(self.name)
    except Exception as exc:
        self.fail('Error checking for resource existence in {0} - {1}'.format(self.name, str(exc)))
    for item in response:
        found = True
        break
    return found