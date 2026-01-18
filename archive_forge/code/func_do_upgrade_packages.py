from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def do_upgrade_packages(module, full=False):
    if full:
        cmd = 'full-upgrade'
    else:
        cmd = 'upgrade'
    rc, out, err = module.run_command(format_pkgin_command(module, cmd))
    if rc == 0:
        if re.search('^(.*\n|)nothing to do.\n$', out):
            module.exit_json(changed=False, msg='nothing left to upgrade')
    else:
        module.fail_json(msg='could not %s packages' % cmd, stdout=out, stderr=err)