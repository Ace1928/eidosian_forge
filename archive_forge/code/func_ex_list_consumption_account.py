import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_consumption_account(self, from_date: str=None, to_date: str=None, dry_run: bool=False):
    """
        Displays information about the consumption of your account for
        each billable resource within the specified time period.


        :param      from_date: The beginning of the time period, in
        ISO 8601 date-time format (for example, 2017-06-14 or
        2017-06-14T00:00:00Z). (required)
        :type       from_date: ``str``

        :param      to_date: The end of the time period, in
        ISO 8601 date-time format (for example, 2017-06-30 or
        2017-06-30T00:00:00Z). (required)
        :type       to_date: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of Consumption Entries
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadConsumptionAccount'
    data = {'DryRun': dry_run}
    if from_date is not None:
        data.update({'FromDate': from_date})
    if to_date is not None:
        data.update({'ToDate': to_date})
    response = self._call_api(action, json.dumps(data))
    print(response.status_code)
    if response.status_code == 200:
        return response.json()['ConsumptionEntries']
    return response.json()