from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def preprocess_read_response(resp):
    v = resp.get('os-extended-volumes:volumes_attached')
    if v and isinstance(v, list):
        for i in range(len(v)):
            if v[i].get('bootIndex') == '0':
                root_volume = v[i]
                if i + 1 != len(v):
                    v[i] = v[-1]
                v.pop()
                resp['root_volume'] = root_volume
                break
    v = resp.get('addresses')
    if v:
        rv = {}
        eips = []
        for val in v.values():
            for item in val:
                if item['OS-EXT-IPS:type'] == 'floating':
                    eips.append(item)
                else:
                    rv[item['OS-EXT-IPS:port_id']] = item
        for item in eips:
            k = item['OS-EXT-IPS:port_id']
            if k in rv:
                rv[k]['eip_address'] = item.get('addr', '')
            else:
                rv[k] = item
                item['eip_address'] = item.get('addr', '')
                item['addr'] = ''
        resp['address'] = rv.values()