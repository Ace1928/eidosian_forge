from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def derive_config_from_merged_cmd(command, exist_conf, test_keys=None):
    if not command:
        return exist_conf
    if isinstance(command, list) and isinstance(exist_conf, list):
        nu, new_conf_dict = derive_config_from_merged_cmd_dict({'config': command}, {'config': exist_conf}, test_keys)
        new_conf = new_conf_dict.get('config', [])
    elif isinstance(command, dict) and isinstance(exist_conf, dict):
        merge_op_dft = get_merge_op('__default_ops', test_keys)
        nu, new_conf = derive_config_from_merged_cmd_dict(command, exist_conf, test_keys, None, None, merge_op_dft)
    elif isinstance(command, dict) and isinstance(exist_conf, list):
        nu, new_conf_dict = derive_config_from_merged_cmd_dict({'config': [command]}, {'config': exist_conf}, test_keys)
        new_conf = new_conf_dict.get('config', [])
    else:
        new_conf = exist_conf
    return new_conf