from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_subscription_payload():
    payload = {'Destination': 'https://192.168.1.100:8188', 'EventFormatType': 'MetricReport', 'Context': 'RedfishEvent', 'Protocol': 'Redfish', 'EventTypes': ['MetricReport'], 'SubscriptionType': 'RedfishEvent'}
    return payload