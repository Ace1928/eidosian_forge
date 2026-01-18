import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_accept_net_peering(self, net_peering_id: List[str]=None, dry_run: bool=False):
    """
        Accepts a Net peering connection request.
        To accept this request, you must be the owner of the peer Net. If
        you do not accept the request within 7 days, the state of the Net
        peering connection becomes expired.

        :param      net_peering_id: The ID of the Net peering connection you
        want to accept. (required)
        :type       net_peering_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The accepted Net Peering
        :rtype: ``dict``
        """
    action = 'AcceptNetPeering'
    data = {'DryRun': dry_run}
    if net_peering_id is not None:
        data.update({'NetPeeringId': net_peering_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetPeering']
    return response.json()