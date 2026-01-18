from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def get_current_member_of_group(rest_obj, group_id):
    group_device = rest_obj.get_all_report_details('{0}({1})/Devices'.format(GROUP_URI, group_id))
    device_id_list = [each['Id'] for each in group_device['report_list']]
    return device_id_list