from __future__ import (absolute_import, division, print_function)
import os
import json
import time
from ssl import SSLError
from xml.etree import ElementTree as ET
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def handle_HTTP_error(module, httperr):
    err_message = json.load(httperr)
    err_list = err_message.get('error', {}).get('@Message.ExtendedInfo', [{'Message': EXIT_MESSAGE}])
    if err_list:
        err_reason = err_list[0].get('Message', EXIT_MESSAGE)
        if IDEM_MSG_ID in err_list[0].get('MessageId'):
            module.exit_json(msg=err_reason)
    if 'error' in err_message:
        module.fail_json(msg=err_message)