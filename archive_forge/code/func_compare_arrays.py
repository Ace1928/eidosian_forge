from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def compare_arrays(old_params, new_params, param_name):
    """Compare two arrays, including any nested properties on elements."""
    old = old_params.get(param_name, [])
    new = new_params.get(param_name, [])
    if old == [] and new == []:
        return True
    oldd = array_to_dict(old)
    newd = array_to_dict(new)
    newd = dict_merge(oldd, newd)
    return newd == oldd