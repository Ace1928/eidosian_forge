import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_net_access_point(self, net_id: str=None, route_table_ids: List[str]=None, service_name: str=None, dry_run: bool=False):
    """
        Creates a Net access point to access a 3DS OUTSCALE service from this
        Net without using the Internet and External IP addresses.
        You specify the service using its prefix list name. For more
        information, see DescribePrefixLists:
        https://docs.outscale.com/api#describeprefixlists
        To control the routing of traffic between the Net and the specified
        service, you can specify one or more route tables. Virtual machines
        placed in Subnets associated with the specified route table thus use
        the Net access point to access the service. When you specify a route
        table, a route is automatically added to it with the destination set
        to the prefix list ID of the service, and the target set to the ID of
        the access point.

        :param      net_id: The ID of the Net. (required)
        :type       net_id: ``str``

        :param      route_table_ids: One or more IDs of route tables to use
        for the connection.
        :type       route_table_ids: ``list`` of ``str``

        :param      service_name: The prefix list name corresponding to the
        service (for example, com.outscale.eu-west-2.osu for OSU). (required)
        :type       service_name: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Access Net Point
        :rtype: ``dict``
        """
    action = 'CreateNetAccessPoint'
    data = {'DryRun': dry_run}
    if net_id is not None:
        data.update({'NetId': net_id})
    if route_table_ids is not None:
        data.update({'RouteTableIds': route_table_ids})
    if service_name is not None:
        data.update({'ServiceName': service_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetAccessPoint']
    return response.json()