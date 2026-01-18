from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_loadbalancerpool_present(self, lb_id, alias, location, method, persistence, port):
    """
        Checks to see if a load balancer pool exists and creates one if it does not.
        :param lb_id: The loadbalancer id
        :param alias: The account alias
        :param location: the datacenter the load balancer resides in
        :param method: the load balancing method
        :param persistence: the load balancing persistence type
        :param port: the port that the load balancer will listen on
        :return: (changed, group, pool_id) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
            pool_id: The string id of the load balancer pool
        """
    changed = False
    result = port
    if not lb_id:
        return (changed, None, None)
    pool_id = self._loadbalancerpool_exists(alias=alias, location=location, port=port, lb_id=lb_id)
    if not pool_id:
        if not self.module.check_mode:
            result = self.create_loadbalancerpool(alias=alias, location=location, lb_id=lb_id, method=method, persistence=persistence, port=port)
            pool_id = result.get('id')
        changed = True
    return (changed, result, pool_id)