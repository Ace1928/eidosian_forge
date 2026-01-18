import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_vpn_connections(self, bgp_asns: List[int]=None, client_gateway_ids: List[str]=None, connection_types: List[str]=None, route_destination_ip_ranges: List[str]=None, states: List[str]=None, static_routes_only: bool=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, dry_run: bool=False):
    """
        Describes one or more VPN connections.

        :param      bgp_asns: The Border Gateway Protocol (BGP) Autonomous
        System Numbers (ASNs) of the connections.
        :type       bgp_asns: ``list`` of ``int``

        :param      client_gateway_ids: The IDs of the client gateways.
        :type       client_gateway_ids: ``list`` of ``str``

        :param      connection_types: The types of the VPN connections (only
        ipsec.1 is supported).
        :type       connection_types: ``list`` of ``str``

        :param      states: The states of the vpn connections
        (pending | available).
        :type       states: ``str``

        :param      route_destination_ip_ranges: The destination IP ranges.
        :type       route_destination_ip_ranges: ``str``

        :param      static_routes_only: If false, the VPN connection uses
        dynamic routing with Border Gateway Protocol (BGP). If true, routing
        is controlled using static routes. For more information about how to
        create and delete static routes, see CreateVpnConnectionRoute:
        https://docs.outscale.com/api#createvpnconnectionroute and
        DeleteVpnConnectionRoute:
        https://docs.outscale.com/api#deletevpnconnectionroute
        :type       static_routes_only: ``bool``

        :param      tag_keys: the keys of the tags associated with the
        subnets.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the
        subnets.
        :type       tag_values: ``list`` of ``str``

        :param      tags: the key/value combination of the tags associated
        with the subnets, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of Subnets
        :rtype: ``list`` of  ``dict``
        """
    action = 'ReadVpnConnections'
    data = {'DryRun': dry_run, 'Filters': {}}
    if bgp_asns is not None:
        data['Filters'].update({'BgpAsns': bgp_asns})
    if client_gateway_ids is not None:
        data['Filters'].update({'ClientGatewayIds': client_gateway_ids})
    if connection_types is not None:
        data['Filters'].update({'ConnectionTypes': connection_types})
    if states is not None:
        data['Filters'].update({'States': states})
    if route_destination_ip_ranges is not None:
        data['Filters'].update({'RouteDestinationIpRanges': route_destination_ip_ranges})
    if static_routes_only is not None:
        data['Filters'].update({'StaticRoutesOnly': static_routes_only})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['VpnConnections']
    return response.json()