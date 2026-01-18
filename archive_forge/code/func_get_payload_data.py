from __future__ import (absolute_import, division, print_function)
import json
import copy
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def get_payload_data(each, vr_members, vr_id):
    is_change, unsup_media, input_vr_mem = (False, None, {})
    vr_mem = vr_members[each['index'] - 1]
    if each['insert']:
        exist_vr_mem = dict(((k, vr_mem[k]) for k in ['Inserted', 'Image', 'UserName', 'Password'] if vr_mem.get(k) is not None))
        input_vr_mem = {'Inserted': each['insert'], 'Image': each['image']}
        if each['image'].startswith('//') or each['image'].lower().startswith('https://'):
            username, password, domain = (each.get('username'), each.get('password'), each.get('domain'))
            if username is not None:
                if domain is not None:
                    username = '{0}\\{1}'.format(domain, username)
                input_vr_mem['UserName'] = username
            if password is not None:
                input_vr_mem['Password'] = password
        else:
            exist_vr_mem.pop('UserName', None)
            exist_vr_mem.pop('Password', None)
        inp_mt = each.get('media_type')
        if inp_mt is not None and inp_mt == 'CD' and (input_vr_mem['Image'][-4:].lower() != '.iso'):
            unsup_media = each['index']
        if inp_mt is not None and inp_mt == 'DVD' and (input_vr_mem['Image'][-4:].lower() != '.iso'):
            unsup_media = each['index']
        if inp_mt is not None and inp_mt == 'USBStick' and (input_vr_mem['Image'][-4:].lower() != '.img'):
            unsup_media = each['index']
        is_change = bool(set(exist_vr_mem.items()) ^ set(input_vr_mem.items()))
    elif vr_id == 'manager':
        for vr_v in vr_members:
            exist_vr_mem = dict(((k, vr_v[k]) for k in ['Inserted']))
            input_vr_mem = {'Inserted': each.get('insert')}
            is_change = bool(set(exist_vr_mem.items()) ^ set(input_vr_mem.items()))
            if is_change:
                vr_mem = vr_v
                break
    else:
        exist_vr_mem = dict(((k, vr_mem[k]) for k in ['Inserted']))
        input_vr_mem = {'Inserted': each.get('insert')}
        is_change = bool(set(exist_vr_mem.items()) ^ set(input_vr_mem.items()))
    return (is_change, input_vr_mem, vr_mem, unsup_media)