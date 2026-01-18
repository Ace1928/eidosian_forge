from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def attributes_check(module, rest_obj, inp_attr, template_id):
    diff = 0
    try:
        resp = rest_obj.invoke_request('GET', TEMPLATE_ATTRIBUTES.format(template_id=template_id))
        attr_dtls = resp.json_data
        disp_adv_list = inp_attr.get('Attributes', {})
        adv_list = []
        for attr in disp_adv_list:
            if attr.get('DisplayName'):
                split_k = str(attr.get('DisplayName')).split(SEPRTR)
                trimmed = map(str.strip, split_k)
                n_k = SEPRTR.join(trimmed)
                adv_list.append(n_k)
        attr_detailed, attr_map = get_subattr_all(attr_dtls.get('AttributeGroups'), adv_list)
        payload_attr = inp_attr.get('Attributes', [])
        rem_attrs = []
        for attr in payload_attr:
            if attr.get('DisplayName'):
                split_k = str(attr.get('DisplayName')).split(SEPRTR)
                trimmed = map(str.strip, split_k)
                n_k = SEPRTR.join(trimmed)
                id = attr_detailed.get(n_k, '')
                attr['Id'] = id
                attr.pop('DisplayName', None)
            else:
                id = attr.get('Id')
            if id:
                ex_val = attr_map.get(id, {})
                if not ex_val:
                    rem_attrs.append(attr)
                    continue
                if attr.get('Value') != ex_val.get('Value') or attr.get('IsIgnored') != ex_val.get('IsIgnored'):
                    diff = diff + 1
        for rem in rem_attrs:
            payload_attr.remove(rem)
    except Exception:
        diff = 1
    return diff