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
def diffparam_device(self):
    before = [':'.join([i['pathonhost'], i['pathincontainer']]) for i in self.info['hostconfig']['devices']]
    if not before and 'createcommand' in self.info['config']:
        before = [i.lower() for i in self._createcommand('--device')]
    before = [':'.join((i, i)) if len(i.split(':')) == 1 else i for i in before]
    after = [':'.join(i.split(':')[:2]) for i in self.params['device']]
    after = [':'.join((i, i)) if len(i.split(':')) == 1 else i for i in after]
    before, after = ([i.lower() for i in before], [i.lower() for i in after])
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('devices', before, after)