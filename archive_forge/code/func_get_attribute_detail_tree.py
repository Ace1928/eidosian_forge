from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_attribute_detail_tree(rest_obj, prof_id):
    try:
        resp = rest_obj.invoke_request('GET', '{0}({1})/AttributeDetails'.format(PROFILE_VIEW, prof_id))
        attr_list = resp.json_data.get('AttributeGroups')
        attr_detailed, attr_map = get_subattr_all(attr_list)
    except Exception:
        attr_detailed, attr_map = ({}, {})
    return (attr_detailed, attr_map)