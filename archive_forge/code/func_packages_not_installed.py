from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def packages_not_installed(module, names):
    """ Check if each package is installed and return list of the ones absent """
    pkgs = []
    for pkg in names:
        rc, out, err = run_command(module, ['pkginfo', '-q', pkg])
        if rc != 0:
            pkgs.append(pkg)
    return pkgs