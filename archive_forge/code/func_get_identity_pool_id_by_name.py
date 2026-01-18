from __future__ import (absolute_import, division, print_function)
import re
import json
import codecs
import binascii
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_identity_pool_id_by_name(pool_name, rest_obj):
    pool_id = 0
    attributes = None
    identity_list = rest_obj.get_all_report_details(IDENTITY_URI)['report_list']
    for item in identity_list:
        if pool_name == item['Name']:
            pool_id = item['Id']
            attributes = item
            break
    return (pool_id, attributes)