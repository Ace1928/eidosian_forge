from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_vm_prop(module, uuid, prop):
    cmd = [module.vmadm, 'lookup', '-j', '-o', prop, 'uuid={0}'.format(uuid)]
    rc, stdout, stderr = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='Could not perform lookup of {0} on {1}'.format(prop, uuid), exception=stderr)
    try:
        stdout_json = json.loads(stdout)
    except Exception as e:
        module.fail_json(msg='Invalid JSON returned by vmadm for uuid lookup of {0}'.format(prop), details=to_native(e), exception=traceback.format_exc())
    if stdout_json:
        return stdout_json[0].get(prop)