from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def list_containers_raw(self):
    args = self.get_base_args() + ['ps', '--format', 'json', '--all']
    if self.compose_version >= LooseVersion('2.23.0'):
        args.append('--no-trunc')
    kwargs = dict(cwd=self.project_src, check_rc=True)
    if self.compose_version >= LooseVersion('2.21.0'):
        dummy, containers, dummy = self.client.call_cli_json_stream(*args, **kwargs)
    else:
        dummy, containers, dummy = self.client.call_cli_json(*args, **kwargs)
    return containers