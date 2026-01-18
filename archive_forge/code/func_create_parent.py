from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def create_parent(rest_obj, module, static_root):
    try:
        prt = static_root
        payload = {}
        payload['MembershipTypeId'] = 12
        payload['Name'] = module.params.get('parent_group_name')
        payload['ParentId'] = prt['Id']
        prt_resp = rest_obj.invoke_request('POST', OP_URI.format(op='Create'), data={'GroupModel': payload})
        return int(prt_resp.json_data)
    except Exception:
        return static_root['Id']