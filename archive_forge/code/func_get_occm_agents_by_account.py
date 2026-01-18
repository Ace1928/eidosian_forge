from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def get_occm_agents_by_account(rest_api, account_id):
    """
        Collect a list of agents matching account_id.
        :return: list of agents, error
        """
    params = {'account_id': account_id}
    api = '/agents-mgmt/agent'
    headers = {'X-User-Token': rest_api.token_type + ' ' + rest_api.token}
    agents, error, dummy = rest_api.get(api, header=headers, params=params)
    return (agents, error)