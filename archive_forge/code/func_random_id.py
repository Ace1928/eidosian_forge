from __future__ import absolute_import, division, print_function
import time
import json
import random
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
def random_id(self):
    random_id = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz')) + ''.join((random.choice('abcdefghijklmnopqrstuvwxyz1234567890') for key in range(7)))
    return random_id