import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_check_account(self, login: str, password: str, dry_run: bool=False):
    """
        Validates the authenticity of the account.

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :param      login: the login of the account
        :type       login: ``str``

        :param      password: the password of the account
        :type       password: ``str``

        :param      dry_run: the password of the account
        :type       dry_run: ``bool``

        :return: True if the action successful
        :rtype: ``bool``
        """
    action = 'CheckAuthentication'
    data = {'DryRun': dry_run, 'Login': login, 'Password': password}
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()