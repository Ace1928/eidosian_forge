from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def remove_dynamic(entries):
    result = []
    for entry in entries:
        if entry.get('dynamic', False) or entry.get('builtin', False):
            continue
        result.append(entry)
    return result