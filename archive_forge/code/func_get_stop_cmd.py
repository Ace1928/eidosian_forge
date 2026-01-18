from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.validation import check_type_int
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
def get_stop_cmd(self, dry_run):
    args = self.get_base_args() + ['stop']
    if self.timeout is not None:
        args.extend(['--timeout', '%d' % self.timeout])
    if dry_run:
        args.append('--dry-run')
    for service in self.services:
        args.append(service)
    args.append('--')
    return args