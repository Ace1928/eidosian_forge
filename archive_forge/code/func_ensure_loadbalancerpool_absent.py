from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_loadbalancerpool_absent(self, alias, location, name, port):
    """
        Checks to see if a load balancer pool exists and deletes it if it does
        :param alias: The account alias
        :param location: the datacenter the load balancer resides in
        :param name: the name of the load balancer
        :param port: the port that the load balancer listens on
        :return: (changed, result) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
        """
    changed = False
    result = None
    lb_exists = self._loadbalancer_exists(name=name)
    if lb_exists:
        lb_id = self._get_loadbalancer_id(name=name)
        pool_id = self._loadbalancerpool_exists(alias=alias, location=location, port=port, lb_id=lb_id)
        if pool_id:
            changed = True
            if not self.module.check_mode:
                result = self.delete_loadbalancerpool(alias=alias, location=location, lb_id=lb_id, pool_id=pool_id)
        else:
            result = "Pool doesn't exist"
    else:
        result = "LB Doesn't Exist"
    return (changed, result)