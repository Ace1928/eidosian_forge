from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def get_all_nodes_list(self, output='short'):
    """
        Returns list of nodes registered in Swarm

        :param output: Defines format of returned data
        :return:
            If 'output' is 'short' then return data is list of nodes hostnames registered in Swarm,
            if 'output' is 'long' then returns data is list of dict containing the attributes as in
            output of command 'docker node ls'
        """
    nodes_list = []
    nodes_inspect = self.get_all_nodes_inspect()
    if nodes_inspect is None:
        return None
    if output == 'short':
        for node in nodes_inspect:
            nodes_list.append(node['Description']['Hostname'])
    elif output == 'long':
        for node in nodes_inspect:
            node_property = {}
            node_property.update({'ID': node['ID']})
            node_property.update({'Hostname': node['Description']['Hostname']})
            node_property.update({'Status': node['Status']['State']})
            node_property.update({'Availability': node['Spec']['Availability']})
            if 'ManagerStatus' in node:
                if node['ManagerStatus']['Leader'] is True:
                    node_property.update({'Leader': True})
                node_property.update({'ManagerStatus': node['ManagerStatus']['Reachability']})
            node_property.update({'EngineVersion': node['Description']['Engine']['EngineVersion']})
            nodes_list.append(node_property)
    else:
        return None
    return nodes_list