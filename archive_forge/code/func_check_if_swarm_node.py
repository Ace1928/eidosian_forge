from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def check_if_swarm_node(self, node_id=None):
    """
        Checking if host is part of Docker Swarm. If 'node_id' is not provided it reads the Docker host
        system information looking if specific key in output exists. If 'node_id' is provided then it tries to
        read node information assuming it is run on Swarm manager. The get_node_inspect() method handles exception if
        it is not executed on Swarm manager

        :param node_id: Node identifier
        :return:
            bool: True if node is part of Swarm, False otherwise
        """
    if node_id is None:
        try:
            info = self.info()
        except APIError:
            self.fail('Failed to get host information.')
        if info:
            json_str = json.dumps(info, ensure_ascii=False)
            swarm_info = json.loads(json_str)
            if swarm_info['Swarm']['NodeID']:
                return True
            if swarm_info['Swarm']['LocalNodeState'] in ('active', 'pending', 'locked'):
                return True
        return False
    else:
        try:
            node_info = self.get_node_inspect(node_id=node_id)
        except APIError:
            return
        if node_info['ID'] is not None:
            return True
        return False