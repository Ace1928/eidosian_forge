from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def return_result(self, ch_status=False, status=True):
    if not status:
        self.module.fail_json(msg=self.result['message'])
    else:
        self.module.exit_json(changed=ch_status, msg=self.result['message'])