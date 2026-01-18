import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_nets_access_point_services(self, service_ids: List[str]=None, service_names: List[str]=None, dry_run: bool=False):
    """
        Describes 3DS OUTSCALE services available to create Net access points.
        For more information, see CreateNetAccessPoint:
        https://docs.outscale.com/api#createnetaccesspoint

        :param      service_ids: The IDs of the services.
        :type       service_ids: ``list`` of ``str``

        :param      service_names: The names of the prefix lists, which
        identify the 3DS OUTSCALE services they are associated with.
        :type       service_names: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: A list of Services
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadNetAccessPointServices'
    data = {'DryRun': dry_run, 'Filters': {}}
    if service_names is not None:
        data['Filters'].update({'ServiceNames': service_names})
    if service_ids is not None:
        data['Filters'].update({'ServiceIds': service_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Services']
    return response.json()