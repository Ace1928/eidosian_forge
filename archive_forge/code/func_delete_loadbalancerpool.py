from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def delete_loadbalancerpool(self, alias, location, lb_id, pool_id):
    """
        Delete the pool on the provided load balancer
        :param alias: The account alias
        :param location: the datacenter the load balancer resides in
        :param lb_id: the id string of the load balancer
        :param pool_id: the id string of the load balancer pool
        :return: result: The result from the delete API call
        """
    result = None
    try:
        result = self.clc.v2.API.Call('DELETE', '/v2/sharedLoadBalancers/%s/%s/%s/pools/%s' % (alias, location, lb_id, pool_id))
    except APIFailedResponse as e:
        self.module.fail_json(msg='Unable to delete pool for load balancer id "{0}". {1}'.format(lb_id, str(e.response_text)))
    return result