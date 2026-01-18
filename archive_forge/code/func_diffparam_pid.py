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
def diffparam_pid(self):

    def get_container_id_by_name(name):
        rc, podman_inspect_info, err = self.module.run_command([self.module.params['executable'], 'inspect', name, '-f', '{{.Id}}'])
        if rc != 0:
            return None
        return podman_inspect_info.strip()
    before = self.info['hostconfig']['pidmode']
    after = self.params['pid']
    if after is not None and 'container:' in after and ('container:' in before):
        if after.split(':')[1] == before.split(':')[1]:
            return self._diff_update_and_compare('pid', before, after)
        after = 'container:' + get_container_id_by_name(after.split(':')[1])
    return self._diff_update_and_compare('pid', before, after)