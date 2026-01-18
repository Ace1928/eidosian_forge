from __future__ import absolute_import, division, print_function
import os
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native
def cleanup_packages(module, packages):
    p = module.params
    if packages:
        for package in packages:
            if query_package(module, package, 'unmerge'):
                break
        else:
            module.exit_json(changed=False, msg='Packages already absent.')
    args = ['--depclean']
    for flag in ['quiet', 'verbose']:
        if p[flag]:
            args.append('--%s' % flag)
    cmd, (rc, out, err) = run_emerge(module, packages, *args)
    if rc != 0:
        module.fail_json(cmd=cmd, rc=rc, stdout=out, stderr=err)
    removed = 0
    for line in out.splitlines():
        if not line.startswith('Number removed:'):
            continue
        parts = line.split(':')
        removed = int(parts[1].strip())
    changed = removed > 0
    module.exit_json(changed=changed, cmd=cmd, rc=rc, stdout=out, stderr=err, msg='Depclean completed.')