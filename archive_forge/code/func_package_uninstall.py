from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def package_uninstall(module, pkgs):
    cmd = ['pkgutil']
    if module.check_mode:
        cmd.append('-n')
    cmd.append('-ry')
    cmd.extend(pkgs)
    return run_command(module, cmd)