from __future__ import absolute_import, division, print_function
from itertools import product
from ansible.module_utils.basic import AnsibleModule
def run_zfs_raw(self, subcommand=None, args=None):
    """ Run a raw zfs command, fail on error.
        """
    cmd = [self.zfs_path, subcommand or self.subcommand] + (args or []) + [self.name]
    rc, out, err = self.module.run_command(cmd)
    if rc:
        self.module.fail_json(msg='Command `%s` failed: %s' % (' '.join(cmd), err))
    return out