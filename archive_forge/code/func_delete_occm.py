from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def delete_occm(self, rest_api, client_id):
    """
        delete occm
        """
    api = '/agents-mgmt/agent/' + rest_api.format_client_id(client_id)
    headers = {'X-User-Token': rest_api.token_type + ' ' + rest_api.token, 'X-Tenancy-Account-Id': self.parameters['account_id']}
    occm_status, error, dummy = rest_api.delete(api, None, header=headers)
    return (occm_status, error)