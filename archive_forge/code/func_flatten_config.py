from __future__ import absolute_import, division, print_function
from functools import total_ordering
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def flatten_config(data, context):
    """Flatten different contexts in
        the running-config for easier parsing.
    :param data: dict
    :param context: str
    :returns: flattened running config
    """
    data = data.split('\n')
    in_cxt = False
    cur = {}
    for index, x in enumerate(data):
        cur_indent = len(x) - len(x.lstrip())
        if x.strip().startswith(context):
            in_cxt = True
            cur['context'] = x
            cur['indent'] = cur_indent
        elif cur and cur_indent <= cur['indent']:
            in_cxt = False
        elif in_cxt:
            data[index] = cur['context'] + ' ' + x.strip()
    return '\n'.join(data)