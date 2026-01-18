from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def package_list(self):
    """Takes the input package list and resolves packages groups to their package list using the inventory,
        extracts package names from packages given as files or URLs using calls to pacman

        Returns the expanded/resolved list as a list of Package
        """
    pkg_list = []
    for pkg in self.m.params['name']:
        if not pkg:
            continue
        is_URL = False
        if pkg in self.inventory['available_groups']:
            for group_member in self.inventory['available_groups'][pkg]:
                pkg_list.append(Package(name=group_member, source=group_member))
        elif pkg in self.inventory['available_pkgs'] or pkg in self.inventory['installed_pkgs']:
            pkg_list.append(Package(name=pkg, source=pkg))
        else:
            cmd = [self.pacman_path, '--sync', '--print-format', '%n', pkg]
            rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
            if rc != 0:
                cmd = [self.pacman_path, '--upgrade', '--print-format', '%n', pkg]
                rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
                if rc != 0:
                    if self.target_state == 'absent':
                        continue
                    else:
                        self.fail(msg='Failed to list package %s' % pkg, cmd=cmd, stdout=stdout, stderr=stderr, rc=rc)
                stdout = stdout.splitlines()[-1]
                is_URL = True
            pkg_name = stdout.strip()
            pkg_list.append(Package(name=pkg_name, source=pkg, source_is_URL=is_URL))
    return pkg_list