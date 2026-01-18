from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def remove_packages(module, xbps_path, packages):
    """Returns true if package removal succeeds"""
    changed_packages = []
    for package in packages:
        installed, updated = query_package(module, xbps_path, package)
        if not installed:
            continue
        cmd = '%s -y %s' % (xbps_path['remove'], package)
        rc, stdout, stderr = module.run_command(cmd, check_rc=False)
        if rc != 0:
            module.fail_json(msg='failed to remove %s' % package)
        changed_packages.append(package)
    if len(changed_packages) > 0:
        module.exit_json(changed=True, msg='removed %s package(s)' % len(changed_packages), packages=changed_packages)
    module.exit_json(changed=False, msg='package(s) already absent')