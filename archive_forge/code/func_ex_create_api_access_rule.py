import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_api_access_rule(self, description: str=None, ip_ranges: List[str]=None, ca_ids: List[str]=None, cns: List[str]=None, dry_run: bool=False):
    """
        Create an API access rule.
        It is a rule to allow access to the API from your account.
        You need to specify at least the CaIds or the IpRanges parameter.

        :param      description: The description of the new rule.
        :type       description: ``str``

        :param      ip_ranges: One or more IP ranges, in CIDR notation
        (for example, 192.0.2.0/16).
        :type       ip_ranges: ``List`` of ``str``

        :param      ca_ids: One or more IDs of Client Certificate Authorities
        (CAs).
        :type       ca_ids: ``List`` of ``str``

        :param      cns: One or more Client Certificate Common Names (CNs).
        If this parameter is specified, you must also specify the ca_ids
        parameter.
        :type       cns: ``List`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a dict containing the API access rule created.
        :rtype: ``dict``
        """
    if not ca_ids and (not ip_ranges):
        raise ValueError('Either ca_ids or ip_ranges argument must be provided.')
    action = 'CreateApiAccessRule'
    data = {'DryRun': dry_run}
    if description is not None:
        data['Description'] = description
    if ip_ranges is not None:
        data['IpRanges'] = ip_ranges
    if ca_ids is not None:
        data['CaIds'] = ca_ids
    if cns is not None:
        data['Cns'] = cns
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ApiAccessRule']
    return response.json()