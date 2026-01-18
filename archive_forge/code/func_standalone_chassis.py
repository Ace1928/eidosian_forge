from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def standalone_chassis(module, rest_obj):
    key, value = (None, None)
    ipaddress = get_ip_from_host(module.params['hostname'])
    resp = rest_obj.invoke_request('GET', DOMAIN_URI)
    for data in resp.json_data['value']:
        if ipaddress in data['PublicAddress']:
            key, value = ('Id', data['DeviceId'])
            break
    else:
        module.fail_json(msg='Failed to fetch the device information.')
    return (key, value)