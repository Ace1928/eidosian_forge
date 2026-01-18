import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_send_reset_password_email(self, email: str, dry_run: bool=False):
    """
        Replaces the account password with the new one you provide.
        You must also provide the token you received by email when asking for
        a password reset using the SendResetPasswordEmail method.

        :param      email: The email address provided for the account.
        :type       email: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'SendResetPasswordEmail'
    data = json.dumps({'DryRun': dry_run, 'Email': email})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return True
    return response.json()