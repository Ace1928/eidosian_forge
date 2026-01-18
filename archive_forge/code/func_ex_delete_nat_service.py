import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_nat_service(self, nat_service_id: str=None, dry_run: bool=False):
    """
        Deletes a specified network address translation (NAT) service.
        This action disassociates the External IP address (EIP) from the NAT
        service, but does not release this EIP from your account. However, it
        does not delete any NAT service routes in your route tables.

        :param      nat_service_id: the ID of the NAT service you want to
        delete. (required)
        :type       nat_service_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteNatService'
    data = {'DryRun': dry_run}
    if nat_service_id is not None:
        data.update({'NatServiceId': nat_service_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()