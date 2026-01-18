import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_net_peerings(self, accepter_net_account_ids: List[str]=None, accepter_net_ip_ranges: List[str]=None, accepter_net_net_ids: List[str]=None, net_peering_ids: List[str]=None, source_net_account_ids: List[str]=None, source_net_ip_ranges: List[str]=None, source_net_net_ids: List[str]=None, state_messages: List[str]=None, states_names: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, dry_run: bool=False):
    """
        Lists one or more peering connections between two Nets.

        :param      accepter_net_account_ids: The account IDs of the owners of
        the peer Nets.
        :type       accepter_net_account_ids: ``list`` of ``str``

        :param      accepter_net_ip_ranges: The IP ranges of the peer Nets, in
        CIDR notation (for example, 10.0.0.0/24).
        :type       accepter_net_ip_ranges: ``list`` of ``str``

        :param      accepter_net_net_ids: The IDs of the peer Nets.
        :type       accepter_net_net_ids: ``list`` of ``str``

        :param      source_net_account_ids: The account IDs of the owners of
        the peer Nets.
        :type       source_net_account_ids: ``list`` of ``str``

        :param      source_net_ip_ranges: The IP ranges of the peer Nets.
        :type       source_net_ip_ranges: ``list`` of ``str``

        :param      source_net_net_ids: The IDs of the peer Nets.
        :type       source_net_net_ids: ``list`` of ``str``

        :param      net_peering_ids: The IDs of the Net peering connections.
        :type       net_peering_ids: ``list`` of ``str``

        :param      state_messages: Additional information about the states of
        the Net peering connections.
        :type       state_messages: ``list`` of ``str``

        :param      states_names: The states of the Net peering connections
        (pending-acceptance | active | rejected | failed | expired | deleted).
        :type       states_names: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with the Net
        peering connections.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: the values of the tags associated with the
        Net peering connections.
        :type       tag_values: ``list`` of ``str``

        :param      tags: The key/value combination of the tags associated
        with the Net peering connections, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: A list of Net Access Points
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadNetPeerings'
    data = {'DryRun': dry_run, 'Filters': {}}
    if accepter_net_account_ids is not None:
        data['Filters'].update({'AccepterNetAccountIds': accepter_net_account_ids})
    if accepter_net_ip_ranges is not None:
        data['Filters'].update({'AccepterNetIpRanges': accepter_net_ip_ranges})
    if accepter_net_net_ids is not None:
        data['Filters'].update({'AccepterNetNetIds': accepter_net_net_ids})
    if source_net_account_ids is not None:
        data['Filters'].update({'SourceNetAccountIds': source_net_account_ids})
    if source_net_ip_ranges is not None:
        data['Filters'].update({'SourceNetIpRanges': source_net_ip_ranges})
    if source_net_net_ids is not None:
        data['Filters'].update({'SourceNetNetIds': source_net_net_ids})
    if net_peering_ids is not None:
        data['Filters'].update({'NetPeeringIds': net_peering_ids})
    if state_messages is not None:
        data['Filters'].update({'StateMessages': state_messages})
    if states_names is not None:
        data['Filters'].update({'StateNames': states_names})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetPeerings']
    return response.json()