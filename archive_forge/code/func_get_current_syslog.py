from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_current_syslog(rest_obj):
    resp = rest_obj.invoke_request('GET', SYSLOG_GET)
    syslog_list = resp.json_data.get('value')
    return syslog_list