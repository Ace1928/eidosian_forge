import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_read_api_access_rules(self, api_access_rules_ids: List[str]=None, ca_ids: List[str]=None, cns: List[str]=None, descriptions: List[str]=None, ip_ranges: List[str]=None, dry_run: bool=False):
    """
        Read API access rules.

        :param      api_access_rules_ids: The List containing rules ids to
        filter the request.
        :type       api_access_rules_ids: ``List`` of ``str``

        :param      ca_ids: The List containing CA ids to filter the request.
        :type       ca_ids: ``List`` of ``str``

        :param      cns: The List containing cns to filter the request.
        :type       cns: ``List`` of ``str``

        :param      descriptions: The List containing descriptions to filter
        the request.
        :type       descriptions: ``List`` of ``str``

        :param      ip_ranges: The List containing ip ranges in CIDR notation
        (for example, 192.0.2.0/16) to filter the request.
        :type       ip_ranges: ``List`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a List of API access rules.
        :rtype: ``List`` of ``dict`` if successful or  ``dict``
        """
    action = 'ReadApiAccessRules'
    filters = {}
    if api_access_rules_ids is not None:
        filters['ApiAccessRulesIds'] = api_access_rules_ids
    if ca_ids is not None:
        filters['CaIds'] = ca_ids
    if cns is not None:
        filters['Cns'] = cns
    if descriptions is not None:
        filters['Descriptions'] = descriptions
    if ip_ranges is not None:
        filters['IpRanges'] = ip_ranges
    data = {'Filters': filters, 'DryRun': dry_run}
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ApiAccessRules']
    return response.json()