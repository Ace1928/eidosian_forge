from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.common.text.converters import to_native
def get_service_path(module, service):
    rc, out, err = run_sys_ctl(module, ['find', service])
    if rc != 0:
        fail_if_missing(module, False, service, msg='host')
    else:
        return to_native(out).strip()