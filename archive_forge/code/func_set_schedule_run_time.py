from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import time
import json
def set_schedule_run_time(self):
    return time.strftime('%Y-%m-%d', time.gmtime()) + 'T' + self.time + ':00Z'