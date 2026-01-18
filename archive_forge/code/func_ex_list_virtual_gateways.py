import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_virtual_gateways(self, connection_types: List[str]=None, link_net_ids: List[str]=None, link_states: List[str]=None, states: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, virtual_gateway_id: List[str]=None, dry_run: bool=False):
    """
        Lists one or more virtual gateways.

        :param      connection_types: The types of the virtual gateways
        (only ipsec.1 is supported).
        :type       connection_types: ``list`` of ``dict``

        :param      link_net_ids: The IDs of the Nets the virtual gateways
        are attached to.
        :type       link_net_ids: ``list`` of ``dict``

        :param      link_states: The current states of the attachments
        between the virtual gateways and the Nets
        (attaching | attached | detaching | detached).
        :type       link_states: ``list`` of ``dict``

        :param      states: The states of the virtual gateways
        (pending | available | deleting | deleted).
        :type       states: ``list`` of ``dict``

        :param      tag_keys: The keys of the tags associated with the
        virtual gateways.
        :type       tag_keys: ``list`` of ``dict``

        :param      tag_values: The values of the tags associated with
        the virtual gateways.
        :type       tag_values: ``list`` of ``dict``

        :param      tags: The key/value combination of the tags associated
        with the virtual gateways, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``dict``

        :param      virtual_gateway_id: The IDs of the virtual gateways.
        :type       virtual_gateway_id: ``list`` of ``dict``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: list of virtual gateway
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadVirtualGateways'
    data = {'Filters': {}, 'DryRun': dry_run}
    if connection_types is not None:
        data['Filters'].update({'ConnectionTypes': connection_types})
    if link_net_ids is not None:
        data['Filters'].update({'LinkNetIds': link_net_ids})
    if link_states is not None:
        data['Filters'].update({'LinkStates': link_states})
    if states is not None:
        data['Filters'].update({'States': states})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    if virtual_gateway_id is not None:
        data['Filters'].update({'VirtualGatewayIds': virtual_gateway_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['VirtualGateways']
    return response.json()