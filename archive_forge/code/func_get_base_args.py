from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def get_base_args(self):
    args = ['compose', '--ansi', 'never']
    if self.compose_version >= LooseVersion('2.19.0'):
        args.extend(['--progress', 'plain'])
    args.extend(['--project-directory', self.project_src])
    if self.project_name:
        args.extend(['--project-name', self.project_name])
    for file in self.files or []:
        args.extend(['--file', file])
    for env_file in self.env_files or []:
        args.extend(['--env-file', env_file])
    for profile in self.profiles or []:
        args.extend(['--profile', profile])
    return args