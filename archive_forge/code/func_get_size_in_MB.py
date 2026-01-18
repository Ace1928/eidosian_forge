from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_size_in_MB(module, size_str):
    SYMBOLS = ['B', 'KB', 'MB', 'GB', 'TB']
    s = size_str
    init = size_str
    num = ''
    while s and s[0:1].isdigit() or s[0:1] == '.':
        num += s[0]
        s = s[1:]
    num = float(num)
    symbol = s.strip()
    if symbol not in SYMBOLS:
        module.fail_json(msg='Cannot interpret %r %r %d' % (init, symbol, num))
    prefix = {'B': 1}
    for i, s in enumerate(SYMBOLS[1:]):
        prefix[s] = 1 << (i + 1) * 10
    size_in_bytes = int(num * prefix[symbol])
    size_in_MB = size_in_bytes / (1024 * 1024)
    return size_in_MB