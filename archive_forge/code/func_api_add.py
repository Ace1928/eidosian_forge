from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def api_add(self):
    param = self.list_to_dic(self.split_params(self.add))
    try:
        self.result['message'].append('added: .id= %s' % self.api_path.add(**param))
        self.return_result(True)
    except LibRouterosError as e:
        self.errors(e)