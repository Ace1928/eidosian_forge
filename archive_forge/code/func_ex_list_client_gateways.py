import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_client_gateways(self, client_gateway_ids: list=None, bgp_asns: list=None, connection_types: list=None, public_ips: list=None, states: list=None, tag_keys: list=None, tag_values: list=None, tags: list=None, dry_run: bool=False):
    """
        Deletes a client gateway.
        You must delete the VPN connection before deleting the client gateway.

        :param      client_gateway_ids: The IDs of the client gateways.
        you want to delete. (required)
        :type       client_gateway_ids: ``list`` of ``str`

        :param      bgp_asns: The Border Gateway Protocol (BGP) Autonomous
        System Numbers (ASNs) of the connections.
        :type       bgp_asns: ``list`` of ``int``

        :param      connection_types: The types of communication tunnels
        used by the client gateways (only ipsec.1 is supported).
        (required)
        :type       connection_types: ``list```of ``str``

        :param      public_ips: The public IPv4 addresses of the
        client gateways.
        :type       public_ips: ``list`` of ``str``

        :param      states: The states of the client gateways
        (pending | available | deleting | deleted).
        :type       states: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with
        the client gateways.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the
        client gateways.
        :type       tag_values: ``list`` of ``str``

        :param      tags: the key/value combination of the tags
        associated with the client gateways, in the following
        format: "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Returns ``list`` of Client Gateway
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadClientGateways'
    data = {'DryRun': dry_run, 'Filters': {}}
    if client_gateway_ids is not None:
        data['Filters'].update({'ClientGatewayIds': client_gateway_ids})
    if bgp_asns is not None:
        data['Filters'].update({'BgpAsns': bgp_asns})
    if connection_types is not None:
        data['Filters'].update({'ConnectionTypes': connection_types})
    if public_ips is not None:
        data['Filters'].update({'PublicIps': public_ips})
    if client_gateway_ids is not None:
        data['Filters'].update({'ClientGatewayIds': client_gateway_ids})
    if states is not None:
        data['Filters'].update({'States': states})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ClientGateways']
    return response.json()