import traceback
from typing import Optional
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_text
@property
def server_side_dry_run(self):
    return self.check_mode and self.has_at_least('kubernetes', '18.20.0')