from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def get_swarm_node_id(self):
    """
        Get the 'NodeID' of the Swarm node or 'None' if host is not in Swarm. It returns the NodeID
        of Docker host the module is executed on
        :return:
            NodeID of host or 'None' if not part of Swarm
        """
    try:
        info = self.info()
    except APIError as exc:
        self.fail('Failed to get node information for %s' % to_native(exc))
    if info:
        json_str = json.dumps(info, ensure_ascii=False)
        swarm_info = json.loads(json_str)
        if swarm_info['Swarm']['NodeID']:
            return swarm_info['Swarm']['NodeID']
    return None