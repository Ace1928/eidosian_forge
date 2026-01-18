import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_flexible_gpu_catalog(self, dry_run: bool=False):
    """
        Lists all flexible GPUs available in the public catalog.

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Returns the Flexible Gpu Catalog
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadFlexibleGpuCatalog'
    data = {'DryRun': dry_run}
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['FlexibleGpuCatalog']
    return response.json()