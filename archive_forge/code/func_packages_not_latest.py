from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def packages_not_latest(module, names, site, update_catalog):
    """ Check status of each package and return list of the ones with an upgrade available """
    cmd = ['pkgutil']
    if update_catalog:
        cmd.append('-U')
    cmd.append('-c')
    if site is not None:
        cmd.extend(['-t', site])
    if names != ['*']:
        cmd.extend(names)
    rc, out, err = run_command(module, cmd)
    packages = []
    for line in out.split('\n')[1:-1]:
        if 'catalog' not in line and 'SAME' not in line:
            packages.append(line.split(' ')[0])
    return list(set(packages))