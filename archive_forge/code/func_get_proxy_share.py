from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_proxy_share(module):
    proxy_share = {}
    proxy_support = module.params.get('proxy_support')
    proxy_type = module.params['proxy_type']
    proxy_server = module.params.get('proxy_server')
    proxy_port = module.params['proxy_port']
    proxy_username = module.params.get('proxy_username')
    proxy_password = module.params.get('proxy_password')
    if proxy_support is True and proxy_server is None:
        module.fail_json(msg=PROXY_ERR_MSG)
    if proxy_support is True:
        proxy_share['proxy_server'] = proxy_server
        proxy_share['proxy_username'] = proxy_username
        proxy_share['proxy_password'] = proxy_password
        proxy_share['proxy_port'] = proxy_port
        proxy_share['proxy_type'] = proxy_type.upper()
        proxy_share['proxy_support'] = 'Enabled'
    else:
        proxy_share['proxy_support'] = 'Disabled'
    return proxy_share