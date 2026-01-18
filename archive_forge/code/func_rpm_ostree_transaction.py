from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native, to_text
def rpm_ostree_transaction(module):
    cmd = []
    cmd.append(module.get_bin_path('rpm-ostree'))
    cmd.append('upgrade')
    if module.params['os']:
        cmd += ['--os', module.params['os']]
    if module.params['cache_only']:
        cmd += ['--cache-only']
    if module.params['allow_downgrade']:
        cmd += ['--allow-downgrade']
    if module.params['peer']:
        cmd += ['--peer']
    module.run_command_environ_update = dict(LANG='C', LC_ALL='C', LC_MESSAGES='C')
    rc, out, err = module.run_command(cmd)
    if rc != 0:
        module.fail_json(rc=rc, msg=err)
    elif to_text('No upgrade available.') in to_text(out):
        module.exit_json(msg=out, changed=False)
    else:
        module.exit_json(msg=out, changed=True)