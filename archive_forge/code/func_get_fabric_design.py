from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def get_fabric_design(fabric_design_uri, rest_obj):
    """
    Get the fabric design name from the fabric design uri which is returned from GET request
    :param fabric_design_uri: fabric design uri
    :param rest_obj: session object
    :return: dict
    """
    fabric_design = {}
    if fabric_design_uri:
        resp = rest_obj.invoke_request('GET', fabric_design_uri.split('/api/')[-1])
        design_type = resp.json_data.get('Name')
        fabric_design = {'Name': design_type}
    return fabric_design