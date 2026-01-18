from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def check_if_swarm_worker(self):
    """
        Checks if node role is set as Worker in Swarm. The node is the docker host on which module action
        is performed. Will fail if run on host that is not part of Swarm via check_if_swarm_node()

        :return: True if node is Swarm Worker, False otherwise
        """
    if self.check_if_swarm_node() and (not self.check_if_swarm_manager()):
        return True
    return False