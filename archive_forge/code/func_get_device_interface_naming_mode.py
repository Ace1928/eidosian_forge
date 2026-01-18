from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_device_interface_naming_mode(module):
    intf_naming_mode = ''
    request = {'path': 'data/sonic-device-metadata:sonic-device-metadata/DEVICE_METADATA/DEVICE_METADATA_LIST=localhost', 'method': GET}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    if 'sonic-device-metadata:DEVICE_METADATA_LIST' in response[0][1]:
        device_meta_data = response[0][1].get('sonic-device-metadata:DEVICE_METADATA_LIST', [])
        if device_meta_data:
            intf_naming_mode = device_meta_data[0].get('intf_naming_mode', 'native')
    return intf_naming_mode