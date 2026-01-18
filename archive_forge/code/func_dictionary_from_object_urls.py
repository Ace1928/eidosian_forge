from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def dictionary_from_object_urls(self, object_urls):
    objects_by_object_id = {}
    for urls in object_urls:
        object_id = urls.split('/')[-1]
        objects_by_object_id[object_id] = urls
    return objects_by_object_id