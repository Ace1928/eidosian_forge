from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def set_loadbalancernodes(self, alias, location, lb_id, pool_id, nodes):
    """
        Updates nodes to the provided pool
        :param alias: the account alias
        :param location: the datacenter the load balancer resides in
        :param lb_id: the id string of the load balancer
        :param pool_id: the id string of the pool
        :param nodes: a list of dictionaries containing the nodes to set
        :return: result: The result from the CLC API call
        """
    result = None
    if not lb_id:
        return result
    if not self.module.check_mode:
        try:
            result = self.clc.v2.API.Call('PUT', '/v2/sharedLoadBalancers/%s/%s/%s/pools/%s/nodes' % (alias, location, lb_id, pool_id), json.dumps(nodes))
        except APIFailedResponse as e:
            self.module.fail_json(msg='Unable to set nodes for the load balancer pool id "{0}". {1}'.format(pool_id, str(e.response_text)))
    return result