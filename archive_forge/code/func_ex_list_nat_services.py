import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_nat_services(self, nat_service_ids: List[str]=None, net_ids: List[str]=None, states: List[str]=None, subnet_ids: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, dry_run: bool=False):
    """
        Lists one or more network address translation (NAT) services.

        :param      nat_service_ids: The IDs of the NAT services.
        :type       nat_service_ids: ``list`` of ``str``

        :param      net_ids: The IDs of the Nets in which the NAT services are.
        :type       net_ids: ``list`` of ``str``

        :param      states: The states of the NAT services
        (pending | available | deleting | deleted).
        :type       states: ``list`` of ``str``

        :param      subnet_ids: The IDs of the Subnets in which the NAT
        services are.
        :type       subnet_ids: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with the NAT
        services.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the NAT
        services.
        :type       tag_values: ``list`` of ``str``

        :param      tags: The values of the tags associated with the NAT
        services.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of back end vms health
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadNatServices'
    data = {'DryRun': dry_run, 'Filters': {}}
    if nat_service_ids is not None:
        data['Filters'].update({'NatServiceIds': nat_service_ids})
    if states is not None:
        data['Filters'].update({'States': states})
    if net_ids is not None:
        data['Filters'].update({'NetIds': net_ids})
    if subnet_ids is not None:
        data['Filters'].update({'SubnetIds': subnet_ids})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NatServices']
    return response.json()