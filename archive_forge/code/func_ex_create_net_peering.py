import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_net_peering(self, accepter_net_id: str=None, source_net_id: str=None, dry_run: bool=False):
    """
        Requests a Net peering connection between a Net you own and a peer Net
        that belongs to you or another account.
        This action creates a Net peering connection that remains in the
        pending-acceptance state until it is accepted by the owner of the peer
        Net. If the owner of the peer Net does not accept the request within
        7 days, the state of the Net peering connection becomes expired.
        For more information, see AcceptNetPeering:
        https://docs.outscale.com/api#acceptnetpeering

        :param      accepter_net_id: The ID of the Net you want to connect
        with. (required)
        :type       accepter_net_id: ``str``

        :param      source_net_id: The ID of the Net you send the peering
        request from. (required)
        :type       source_net_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Net Peering
        :rtype: ``dict``
        """
    action = 'CreateNetPeering'
    data = {'DryRun': dry_run}
    if accepter_net_id is not None:
        data.update({'AccepterNetId': accepter_net_id})
    if source_net_id is not None:
        data.update({'SourceNetId': source_net_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetPeering']
    return response.json()