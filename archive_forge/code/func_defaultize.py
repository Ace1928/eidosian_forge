from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def defaultize(self):
    params_with_defaults = {}
    self.default_dict = PodmanPodDefaults(self.module, self.version).default_dict()
    for p in self.module_params:
        if self.module_params[p] is None and p in self.default_dict:
            params_with_defaults[p] = self.default_dict[p]
        else:
            params_with_defaults[p] = self.module_params[p]
    return params_with_defaults