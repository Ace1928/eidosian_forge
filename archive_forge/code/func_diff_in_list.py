from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec, validate_ip_v6_address
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_config, load_config
def diff_in_list(want, have):
    adds = set()
    removes = set()
    for w in want:
        if w['dest'] == 'buffered':
            for h in have:
                if h['dest'] == 'buffered':
                    adds = w['level'] - h['level']
                    removes = h['level'] - w['level']
                    return (adds, removes)
    return (adds, removes)