from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_new_config(commands, exist_conf, test_keys=None):
    if not commands:
        return exist_conf
    cmds = deepcopy(commands)
    n_conf = list()
    e_conf = exist_conf
    for cmd in cmds:
        state = cmd['state']
        cmd.pop('state')
        if state == 'merged':
            n_conf = derive_config_from_merged_cmd(cmd, e_conf, test_keys)
        elif state == 'deleted':
            n_conf = derive_config_from_deleted_cmd(cmd, e_conf, test_keys)
        elif state == 'replaced':
            n_conf = derive_config_from_merged_cmd(cmd, e_conf, test_keys)
        elif state == 'overridden':
            n_conf = derive_config_from_merged_cmd(cmd, e_conf, test_keys)
        e_conf = n_conf
    return n_conf