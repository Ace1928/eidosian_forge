from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_ad(module, rest_obj):
    ad = {}
    prm = module.params
    resp = rest_obj.invoke_request('GET', AD_URI)
    ad_list = resp.json_data.get('value')
    ad_cnt = len(ad_list)
    ky = 'Name'
    vl = 'name'
    if prm.get('id'):
        ky = 'Id'
        vl = 'id'
    for adx in ad_list:
        if str(adx.get(ky)).lower() == str(prm.get(vl)).lower():
            ad = adx
            break
    return (ad, ad_cnt)