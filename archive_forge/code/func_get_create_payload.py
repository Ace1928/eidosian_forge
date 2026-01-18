from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_create_payload(module, rest_obj, deviceid, view_id):
    create_payload = {'Fqdds': 'All', 'ViewTypeId': view_id}
    attrib_dict = module.params.get('attributes').copy()
    if isinstance(attrib_dict, dict):
        typeid = attrib_dict.get('Type') if attrib_dict.get('Type') else attrib_dict.get('TypeId')
        if typeid:
            create_payload['TypeId'] = typeid
        attrib_dict.pop('Type', None)
        create_payload.update(attrib_dict)
        template = get_template_by_name(attrib_dict.get('Name'), module, rest_obj)
        if template:
            module.exit_json(msg=TEMPLATE_NAME_EXISTS.format(name=attrib_dict.get('Name')))
    create_payload['SourceDeviceId'] = int(deviceid)
    return create_payload