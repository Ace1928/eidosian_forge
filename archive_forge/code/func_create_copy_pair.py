from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request
def create_copy_pair(params):
    get_status = 'storage-systems/%s/volume-copy-jobs' % params['ssid']
    url = params['api_url'] + get_status
    rData = {'sourceId': params['source_volume_id'], 'targetId': params['destination_volume_id']}
    rc, resp = request(url, data=json.dumps(rData), ignore_errors=True, method='POST', url_username=params['api_username'], url_password=params['api_password'], headers=HEADERS, validate_certs=params['validate_certs'])
    if rc != 200:
        return (False, (rc, resp))
    else:
        return (True, (rc, resp))