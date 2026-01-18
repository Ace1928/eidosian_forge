from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def strip_uplink_info(uplink_info):
    for item in uplink_info:
        item = strip_substr_dict(item)
        if item['Networks']:
            for net in item['Networks']:
                net = strip_substr_dict(net)
        if item['Ports']:
            for port in item['Ports']:
                port = strip_substr_dict(port)
    return uplink_info