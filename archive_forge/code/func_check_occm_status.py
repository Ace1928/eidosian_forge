from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def check_occm_status(rest_api, client_id):
    """
        Check OCCM status
        :return: status
        DEPRECATED - use get_occm_agent_by_id but the retrun value format is different!
        """
    api = '/agents-mgmt/agent/' + rest_api.format_client_id(client_id)
    headers = {'X-User-Token': rest_api.token_type + ' ' + rest_api.token}
    occm_status, error, dummy = rest_api.get(api, header=headers)
    return (occm_status, error)