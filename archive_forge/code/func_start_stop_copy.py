from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request
def start_stop_copy(params):
    get_status = 'storage-systems/%s/volume-copy-jobs-control/%s?control=%s' % (params['ssid'], params['volume_copy_pair_id'], params['start_stop_copy'])
    url = params['api_url'] + get_status
    response_code, response_data = request(url, ignore_errors=True, method='POST', url_username=params['api_username'], url_password=params['api_password'], headers=HEADERS, validate_certs=params['validate_certs'])
    if response_code == 200:
        return (True, response_data[0]['percentComplete'])
    else:
        return (False, response_data)