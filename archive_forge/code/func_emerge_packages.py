from __future__ import absolute_import, division, print_function
import os
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native
def emerge_packages(module, packages):
    """Run emerge command against given list of atoms."""
    p = module.params
    if p['noreplace'] and (not p['changed_use']) and (not p['newuse']) and (not (p['update'] or p['state'] == 'latest')):
        for package in packages:
            if p['noreplace'] and (not p['changed_use']) and (not p['newuse']) and (not query_package(module, package, 'emerge')):
                break
        else:
            module.exit_json(changed=False, msg='Packages already present.')
        if module.check_mode:
            module.exit_json(changed=True, msg='Packages would be installed.')
    args = []
    emerge_flags = {'update': '--update', 'deep': '--deep', 'newuse': '--newuse', 'changed_use': '--changed-use', 'oneshot': '--oneshot', 'noreplace': '--noreplace', 'nodeps': '--nodeps', 'onlydeps': '--onlydeps', 'quiet': '--quiet', 'verbose': '--verbose', 'getbinpkgonly': '--getbinpkgonly', 'getbinpkg': '--getbinpkg', 'usepkgonly': '--usepkgonly', 'usepkg': '--usepkg', 'keepgoing': '--keep-going', 'quietbuild': '--quiet-build', 'quietfail': '--quiet-fail'}
    for flag, arg in emerge_flags.items():
        if p[flag]:
            args.append(arg)
    if p['state'] and p['state'] == 'latest':
        args.append('--update')
    emerge_flags = {'jobs': '--jobs', 'loadavg': '--load-average', 'backtrack': '--backtrack', 'withbdeps': '--with-bdeps'}
    for flag, arg in emerge_flags.items():
        flag_val = p[flag]
        if flag_val is None:
            "Fallback to default: don't use this argument at all."
            continue
        'Add the --flag=value pair.'
        if isinstance(flag_val, bool):
            args.extend((arg, to_native('y' if flag_val else 'n')))
        elif not flag_val:
            'If the value is 0 or 0.0: add the flag, but not the value.'
            args.append(arg)
        else:
            args.extend((arg, to_native(flag_val)))
    cmd, (rc, out, err) = run_emerge(module, packages, *args)
    if rc != 0:
        module.fail_json(cmd=cmd, rc=rc, stdout=out, stderr=err, msg='Packages not installed.')
    if (p['usepkgonly'] or p['getbinpkg'] or p['getbinpkgonly']) and 'Permission denied (publickey).' in err:
        module.fail_json(cmd=cmd, rc=rc, stdout=out, stderr=err, msg='Please check your PORTAGE_BINHOST configuration in make.conf and your SSH authorized_keys file')
    changed = True
    for line in out.splitlines():
        if re.match('(?:>+) Emerging (?:binary )?\\(1 of', line):
            msg = 'Packages installed.'
            break
        elif module.check_mode and re.match('\\[(binary|ebuild)', line):
            msg = 'Packages would be installed.'
            break
    else:
        changed = False
        msg = 'No packages installed.'
    module.exit_json(changed=changed, cmd=cmd, rc=rc, stdout=out, stderr=err, msg=msg)