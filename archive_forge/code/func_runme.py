from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import sysv_is_enabled, get_sysv_script, sysv_exists, fail_if_missing, get_ps, daemonize
def runme(doit):
    args = module.params['arguments']
    cmd = '%s %s %s' % (script, doit, '' if args is None else args)
    if module.params['daemonize']:
        rc, out, err = daemonize(module, cmd)
    else:
        rc, out, err = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='Failed to %s service: %s' % (action, name), rc=rc, stdout=out, stderr=err)
    return (rc, out, err)