from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def register_agent_to_service(self, rest_api, provider, vpc):
    """
        register agent to service
        """
    api = '/agents-mgmt/connector-setup'
    headers = {'X-User-Token': rest_api.token_type + ' ' + rest_api.token}
    body = {'accountId': self.parameters['account_id'], 'name': self.parameters['name'], 'company': self.parameters['company'], 'placement': {'provider': provider, 'region': self.parameters['region'], 'network': vpc, 'subnet': self.parameters['subnet_id']}, 'extra': {'proxy': {'proxyUrl': self.parameters.get('proxy_url'), 'proxyUserName': self.parameters.get('proxy_user_name'), 'proxyPassword': self.parameters.get('proxy_password')}}}
    if provider == 'AWS':
        body['placement']['network'] = vpc
    response, error, dummy = rest_api.post(api, body, header=headers)
    return (response, error)