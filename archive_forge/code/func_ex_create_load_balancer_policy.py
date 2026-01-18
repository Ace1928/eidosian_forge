import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_load_balancer_policy(self, cookie_name: str=None, load_balancer_name: str=None, policy_name: str=None, policy_type: str=None, dry_run: bool=False):
    """
        Creates a stickiness policy with sticky session lifetimes defined by
        the browser lifetime. The created policy can be used with HTTP or
        HTTPS listeners only.
        If this policy is implemented by a load balancer, this load balancer
        uses this cookie in all incoming requests to direct them to the
        specified back-end server virtual machine (VM). If this cookie is not
        present, the load balancer sends the request to any other server
        according to its load-balancing algorithm.

        You can also create a stickiness policy with sticky session lifetimes
        following the lifetime of an application-generated cookie.
        Unlike the other type of stickiness policy, the lifetime of the
        special Load Balancer Unit (LBU) cookie follows the lifetime of the
        application-generated cookie specified in the policy configuration.
        The load balancer inserts a new stickiness cookie only when the
        application response includes a new application cookie.
        The session stops being sticky if the application cookie is removed or
        expires, until a new application cookie is issued.

        :param      cookie_name: The name of the application cookie used for
        stickiness. This parameter is required if you create a stickiness
        policy based on an application-generated cookie.
        :type       cookie_name: ``str``

        :param      load_balancer_name: The name of the load balancer for
        which you want to create a policy. (required)
        :type       load_balancer_name: ``str``

        :param      policy_name: The name of the policy. This name must be
        unique and consist of alphanumeric characters and dashes (-).
        (required)
        :type       policy_name: ``str``

        :param      policy_type: The type of stickiness policy you want to
        create: app or load_balancer. (required)
        :type       policy_type: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Load Balancer Policy
        :rtype: ``dict``
        """
    action = 'CreateLoadBalancerPolicy'
    data = {'DryRun': dry_run, 'Tags': {}}
    if cookie_name is not None:
        data.update({'CookieName': cookie_name})
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    if policy_type is not None:
        data.update({'PolicyType': policy_type})
    if policy_name is not None:
        data.update({'PolicyName': policy_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['LoadBalancer']
    return response.json()