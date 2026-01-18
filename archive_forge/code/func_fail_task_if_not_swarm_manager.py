from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def fail_task_if_not_swarm_manager(self):
    """
        If host is not a swarm manager then Ansible task on this host should end with 'failed' state
        """
    if not self.check_if_swarm_manager():
        self.fail('Error running docker swarm module: must run on swarm manager node')