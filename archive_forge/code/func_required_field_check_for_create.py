from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def required_field_check_for_create(fabric_id, module):
    params = module.params
    if not fabric_id and (not all([params.get('fabric_design'), params.get('primary_switch_service_tag'), params.get('secondary_switch_service_tag')])):
        module.fail_json(msg=REQUIRED_FIELD)