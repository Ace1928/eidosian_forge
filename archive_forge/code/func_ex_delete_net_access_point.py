import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_net_access_point(self, net_access_point_id: str=None, dry_run: bool=False):
    """
        Deletes one or more Net access point.
        This action also deletes the corresponding routes added to the route
        tables you specified for the Net access point.

        :param      net_access_point_id: The ID of the Net access point.
        (required)
        :type       net_access_point_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteNetAccessPoint'
    data = {'DryRun': dry_run}
    if net_access_point_id is not None:
        data.update({'NetAccessPointId': net_access_point_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()