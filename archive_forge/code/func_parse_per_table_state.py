from __future__ import absolute_import, division, print_function
import re
import os
import time
import tempfile
import filecmp
import shutil
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def parse_per_table_state(all_states_dump):
    """
    Convert raw iptables-save output into usable datastructure, for reliable
    comparisons between initial and final states.
    """
    lines = filter_and_format_state(all_states_dump)
    tables = dict()
    current_table = ''
    current_list = list()
    for line in lines:
        if re.match('^[*](filter|mangle|nat|raw|security)$', line):
            current_table = line[1:]
            continue
        if line == 'COMMIT':
            tables[current_table] = current_list
            current_table = ''
            current_list = list()
            continue
        if line.startswith('# '):
            continue
        current_list.append(line)
    return tables