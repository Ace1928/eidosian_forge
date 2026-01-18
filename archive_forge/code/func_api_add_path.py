from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def api_add_path(self, api, path):
    api_path = api.path()
    for p in path:
        api_path = api_path.join(p)
    return api_path