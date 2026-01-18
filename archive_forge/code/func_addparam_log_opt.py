from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
def addparam_log_opt(self, c):
    for k, v in self.params['log_opt'].items():
        if v is not None:
            c += ['--log-opt', b'='.join([to_bytes(k.replace('max_size', 'max-size'), errors='surrogate_or_strict'), to_bytes(v, errors='surrogate_or_strict')])]
    return c