import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_api_access_rule(self, api_access_rule_id: str, dry_run: bool=False):
    """
        Delete an API access rule.
        You cannot delete the last remaining API access rule.

        :param      api_access_rule_id: The id of the targeted rule
        (required).
        :type       api_access_rule_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: true if successful.
        :rtype: ``bool`` if successful or  ``dict``
        """
    action = 'DeleteApiAccessRule'
    data = {'ApiAccessRuleId': api_access_rule_id, 'DryRun': dry_run}
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()