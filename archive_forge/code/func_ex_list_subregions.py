import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_subregions(self, ex_dry_run: bool=False):
    """
        Lists available subregions details.

        :param      ex_dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       ex_dry_run: ``bool``
        :return: subregions details
        :rtype: ``dict``
        """
    action = 'ReadSubregions'
    data = json.dumps({'DryRun': ex_dry_run})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return response.json()['Subregions']
    return response.json()