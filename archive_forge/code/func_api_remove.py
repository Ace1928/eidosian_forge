from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def api_remove(self):
    try:
        self.api_path.remove(self.remove)
        self.result['message'].append('removed: .id= %s' % self.remove)
        self.return_result(True)
    except LibRouterosError as e:
        self.errors(e)