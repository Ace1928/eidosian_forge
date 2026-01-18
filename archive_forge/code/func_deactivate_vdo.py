from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import re
import traceback
def deactivate_vdo(module, vdoname, vdocmd):
    rc, out, err = module.run_command([vdocmd, 'deactivate', '--name=%s' % vdoname])
    if rc == 0:
        module.log('deactivated VDO volume %s' % vdoname)
    return rc