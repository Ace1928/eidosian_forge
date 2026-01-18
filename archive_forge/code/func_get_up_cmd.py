from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.validation import check_type_int
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
def get_up_cmd(self, dry_run, no_start=False):
    args = self.get_base_args() + ['up', '--detach', '--no-color', '--quiet-pull']
    if self.pull != 'policy':
        args.extend(['--pull', self.pull])
    if self.remove_orphans:
        args.append('--remove-orphans')
    if self.recreate == 'always':
        args.append('--force-recreate')
    if self.recreate == 'never':
        args.append('--no-recreate')
    if not self.dependencies:
        args.append('--no-deps')
    if self.timeout is not None:
        args.extend(['--timeout', '%d' % self.timeout])
    if self.build == 'always':
        args.append('--build')
    elif self.build == 'never':
        args.append('--no-build')
    for key, value in sorted(self.scale.items()):
        args.extend(['--scale', '%s=%d' % (key, value)])
    if self.wait:
        args.append('--wait')
        if self.wait_timeout is not None:
            args.extend(['--wait-timeout', str(self.wait_timeout)])
    if no_start:
        args.append('--no-start')
    if dry_run:
        args.append('--dry-run')
    for service in self.services:
        args.append(service)
    args.append('--')
    return args