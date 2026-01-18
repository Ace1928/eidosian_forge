from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def is_valid_reset_type(reset_type, allowable_enum, module):
    if reset_type not in allowable_enum:
        res_list = re.findall('[A-Z][^A-Z]*', reset_type)
        lw_reset_type = ' '.join([word.lower() for word in res_list])
        error_msg = 'The target device does not support a {0} operation.The acceptable values for device reset types are {1}.'.format(lw_reset_type, ', '.join(allowable_enum))
        module.fail_json(msg=error_msg)