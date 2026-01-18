from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_runtime_status(ignore_selinux_state=False):
    return True if ignore_selinux_state is True else selinux.is_selinux_enabled()