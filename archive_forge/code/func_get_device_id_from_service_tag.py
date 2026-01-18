from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def get_device_id_from_service_tag(self, service_tag):
    """
        :param service_tag: service tag of the device
        :return: dict
        Id: int: device id
        value: dict: device id details
        not_found_msg: str: message if service tag not found
        """
    device_id = None
    query = "DeviceServiceTag eq '{0}'".format(service_tag)
    response = self.invoke_request('GET', 'DeviceService/Devices', query_param={'$filter': query})
    value = response.json_data.get('value', [])
    device_info = {}
    if value:
        device_info = value[0]
        device_id = device_info['Id']
    return {'Id': device_id, 'value': device_info}