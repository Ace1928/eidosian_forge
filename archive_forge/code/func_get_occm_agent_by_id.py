from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def get_occm_agent_by_id(rest_api, client_id):
    """
        Fetch OCCM agent given its client id
        :return: agent details, error
        """
    api = '/agents-mgmt/agent/' + rest_api.format_client_id(client_id)
    headers = {'X-User-Token': rest_api.token_type + ' ' + rest_api.token}
    response, error, dummy = rest_api.get(api, header=headers)
    if isinstance(response, dict) and 'agent' in response:
        agent = response['agent']
        return (agent, error)
    return (response, error)