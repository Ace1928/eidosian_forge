from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def install_packages(module, xbps_path, state, packages):
    """Returns true if package install succeeds."""
    toInstall = []
    for i, package in enumerate(packages):
        'If the package is installed and state == present or state == latest\n        and is up-to-date then skip'
        installed, updated = query_package(module, xbps_path, package)
        if installed and (state == 'present' or (state == 'latest' and updated)):
            continue
        toInstall.append(package)
    if len(toInstall) == 0:
        module.exit_json(changed=False, msg='Nothing to Install')
    cmd = '%s -y %s' % (xbps_path['install'], ' '.join(toInstall))
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    if rc == 16 and module.params['upgrade_xbps']:
        upgrade_xbps(module, xbps_path)
        module.params['upgrade_xbps'] = False
        install_packages(module, xbps_path, state, packages)
    elif rc != 0 and (not (state == 'latest' and rc == 17)):
        module.fail_json(msg='failed to install %s packages(s)' % len(toInstall), packages=toInstall)
    module.exit_json(changed=True, msg='installed %s package(s)' % len(toInstall), packages=toInstall)