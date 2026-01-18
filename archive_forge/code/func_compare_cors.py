from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def compare_cors(cors1, cors2):
    if len(cors1) != len(cors2):
        return False
    copy2 = copy.copy(cors2)
    for rule1 in cors1:
        matched = False
        for rule2 in copy2:
            if rule1['max_age_in_seconds'] == rule2['max_age_in_seconds'] and set(rule1['allowed_methods']) == set(rule2['allowed_methods']) and (set(rule1['allowed_origins']) == set(rule2['allowed_origins'])) and (set(rule1['allowed_headers']) == set(rule2['allowed_headers'])) and (set(rule1['exposed_headers']) == set(rule2['exposed_headers'])):
                matched = True
                copy2.remove(rule2)
        if not matched:
            return False
    return True