from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def pretty_to_bytes(pretty_val):
    if not pretty_val:
        return pretty_val
    if not pretty_val[0].isdigit():
        return pretty_val
    if not pretty_val[-1].isalpha():
        try:
            pretty_val = int(pretty_val)
        except ValueError:
            try:
                pretty_val = float(pretty_val)
            except ValueError:
                return pretty_val
        return pretty_val
    num_part = []
    for c in pretty_val:
        if not c.isdigit():
            break
        else:
            num_part.append(c)
    num_part = int(''.join(num_part))
    val_in_bytes = None
    if len(pretty_val) >= 2:
        if 'kB' in pretty_val[-2:]:
            val_in_bytes = num_part * 1024
        elif 'MB' in pretty_val[-2:]:
            val_in_bytes = num_part * 1024 * 1024
        elif 'GB' in pretty_val[-2:]:
            val_in_bytes = num_part * 1024 * 1024 * 1024
        elif 'TB' in pretty_val[-2:]:
            val_in_bytes = num_part * 1024 * 1024 * 1024 * 1024
    if not val_in_bytes and 'B' in pretty_val[-1]:
        val_in_bytes = num_part
    if val_in_bytes is not None:
        return val_in_bytes
    else:
        return pretty_val