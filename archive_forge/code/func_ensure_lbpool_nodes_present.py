from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_lbpool_nodes_present(self, alias, location, name, port, nodes):
    """
        Checks to see if the provided list of nodes exist for the pool and add the missing nodes to the pool
        :param alias: The account alias
        :param location: the datacenter the load balancer resides in
        :param name: the name of the load balancer
        :param port: the port that the load balancer will listen on
        :param nodes: the list of nodes to be added
        :return: (changed, result) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
        """
    changed = False
    lb_exists = self._loadbalancer_exists(name=name)
    if lb_exists:
        lb_id = self._get_loadbalancer_id(name=name)
        pool_id = self._loadbalancerpool_exists(alias=alias, location=location, port=port, lb_id=lb_id)
        if pool_id:
            changed, result = self.add_lbpool_nodes(alias=alias, location=location, lb_id=lb_id, pool_id=pool_id, nodes_to_add=nodes)
        else:
            result = "Pool doesn't exist"
    else:
        result = "Load balancer doesn't Exist"
    return (changed, result)