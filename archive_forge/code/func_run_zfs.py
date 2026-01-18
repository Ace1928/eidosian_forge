from __future__ import absolute_import, division, print_function
from itertools import product
from ansible.module_utils.basic import AnsibleModule
def run_zfs(self, args):
    """ Run zfs allow/unallow with appropriate options as per module arguments.
        """
    args = self.recursive_opt + ['-' + self.scope] + args
    if self.perms:
        args.append(','.join(self.perms))
    return self.run_zfs_raw(args=args)