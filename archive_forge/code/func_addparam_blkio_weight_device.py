from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def addparam_blkio_weight_device(self, c):
    self.check_version('--blkio-weight-device', minv='4.3.0')
    for dev in self.params['blkio_weight_device']:
        c += ['--blkio-weight-device', dev]
    return c