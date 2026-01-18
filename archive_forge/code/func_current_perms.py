from __future__ import absolute_import, division, print_function
from itertools import product
from ansible.module_utils.basic import AnsibleModule
@property
def current_perms(self):
    """ Parse the output of `zfs allow <name>` to retrieve current permissions.
        """
    out = self.run_zfs_raw(subcommand='allow')
    perms = {'l': {'u': {}, 'g': {}, 'e': []}, 'd': {'u': {}, 'g': {}, 'e': []}, 'ld': {'u': {}, 'g': {}, 'e': []}}
    linemap = {'Local permissions:': 'l', 'Descendent permissions:': 'd', 'Local+Descendent permissions:': 'ld'}
    scope = None
    for line in out.splitlines():
        scope = linemap.get(line, scope)
        if not scope:
            continue
        if ' (unknown: ' in line:
            line = line.replace('(unknown: ', '', 1)
            line = line.replace(')', '', 1)
        try:
            if line.startswith('\tuser ') or line.startswith('\tgroup '):
                ent_type, ent, cur_perms = line.split()
                perms[scope][ent_type[0]][ent] = cur_perms.split(',')
            elif line.startswith('\teveryone '):
                perms[scope]['e'] = line.split()[1].split(',')
        except ValueError:
            self.module.fail_json(msg="Cannot parse user/group permission output by `zfs allow`: '%s'" % line)
    return perms