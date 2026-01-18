import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_nets(self, dhcp_options_set_ids: List[str]=None, ip_ranges: List[str]=None, is_default: bool=None, net_ids: List[str]=None, states: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, dry_run: bool=False):
    """
        Lists one or more Nets.

        :param      dhcp_options_set_ids: The IDs of the DHCP options sets.
        :type       dhcp_options_set_ids: ``list`` of ``str``

        :param      ip_ranges: The IP ranges for the Nets, in CIDR notation
        (for example, 10.0.0.0/16).
        :type       ip_ranges: ``list`` of ``str``

        :param      is_default: If true, the Net used is the default one.
        :type       is_default: ``bool``

        :param      net_ids: The IDs of the Nets.
        :type       net_ids: ``list`` of ``str``

        :param      states: The states of the Nets (pending | available).
        :type       states: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with the Nets.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the
        Nets.
        :type       tag_values: ``list`` of ``str``

        :param      tags: The key/value combination of the tags associated
        with the Nets, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: A list of Nets
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadNets'
    data = {'DryRun': dry_run, 'Filters': {}}
    if dhcp_options_set_ids is not None:
        data['Filters'].update({'DhcpOptionsSetIds': dhcp_options_set_ids})
    if ip_ranges is not None:
        data['Filters'].update({'IpRanges': ip_ranges})
    if is_default is not None:
        data['Filters'].update({'IsDefault': is_default})
    if net_ids is not None:
        data['Filters'].update({'NetIds': net_ids})
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
        return response.json()['Nets']
    return response.json()