from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
def get_docker_swarm_unlock_key(self):
    unlock_key = self.client.get_unlock_key() or {}
    return unlock_key.get('UnlockKey') or None