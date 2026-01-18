from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def compare_lists(self, list1, list2):
    if len(list1) != len(list2):
        return False
    for element in list1:
        if element not in list2:
            return False
    return True