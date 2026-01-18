from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import re
import traceback
def start_vdo(module, vdoname, vdocmd):
    rc, out, err = module.run_command([vdocmd, 'start', '--name=%s' % vdoname])
    if rc == 0:
        module.log('started VDO volume %s' % vdoname)
    return rc