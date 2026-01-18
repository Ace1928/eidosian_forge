from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def convert_dict_to_single_entry_list(base_data, compare_with_data, test_keys):
    new_base = {'config': [base_data]}
    new_compare = {'config': [compare_with_data]}
    config_testkey = None
    for item in test_keys:
        for key, val in item.items():
            if key == 'config':
                config_testkey = list(val)[0]
                break
        if config_testkey:
            break
    if config_testkey and base_data and (config_testkey not in base_data):
        new_base = {'config': [{config_testkey: 'temp_key', 'data': base_data}]}
        new_compare = {'config': [{config_testkey: 'temp_key', 'data': compare_with_data}]}
    return (new_base, new_compare)