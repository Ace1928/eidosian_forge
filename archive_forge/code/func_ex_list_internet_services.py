import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_internet_services(self, internet_service_ids: List[str]=None, link_net_ids: List[str]=None, link_states: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, dry_run: bool=False):
    """
        Lists one or more of your Internet services.
        An Internet service enables your virtual machines (VMs) launched in a
        Net to connect to the Internet. By default, a Net includes an
        Internet service, and each Subnet is public. Every VM launched within
        a default Subnet has a private and a public IP addresses.

        :param      internet_service_ids: One or more filters.
        :type       internet_service_ids: ``list`` of ``str``

        :param      link_net_ids: The IDs of the Nets the Internet services
        are attached to.
        :type       link_net_ids: ``list`` of ``str``

        :param      link_states: The current states of the attachments
        between the Internet services and the Nets (only available,
        if the Internet gateway is attached to a VPC). (required)
        :type       link_states: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with the
        Internet services.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the
        Internet services.
        :type       tag_values: ``list`` of ``str``

        :param      tags: The key/value combination of the tags associated
        with the Internet services, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Returns the list of Internet Services
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadInternetServices'
    data = {'DryRun': dry_run, 'Filters': {}}
    if internet_service_ids is not None:
        data['Filters'].update({'InternetServiceIds': internet_service_ids})
    if link_net_ids is not None:
        data['Filters'].update({'LinkNetIds': link_net_ids})
    if link_states is not None:
        data['Filters'].update({'LinkStates': link_states})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['InternetServices']
    return response.json()