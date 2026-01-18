from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def diff_config(commands, config):
    config = [str(c).replace("'", '') for c in config.splitlines()]
    updates = list()
    visited = set()
    for line in commands:
        item = str(line).replace("'", '')
        if not item.startswith('set') and (not item.startswith('delete')):
            raise ValueError('line must start with either `set` or `delete`')
        elif item.startswith('set') and item not in config:
            updates.append(line)
        elif item.startswith('delete'):
            if not config:
                updates.append(line)
            else:
                item = re.sub('delete', 'set', item)
                for entry in config:
                    if entry.startswith(item) and line not in visited:
                        updates.append(line)
                        visited.add(line)
    return list(updates)