from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request
def check_copy_status(params):
    get_status = 'storage-systems/%s/volume-copy-jobs-control/%s' % (params['ssid'], params['volume_copy_pair_id'])
    url = params['api_url'] + get_status
    response_code, response_data = request(url, ignore_errors=True, method='GET', url_username=params['api_username'], url_password=params['api_password'], headers=HEADERS, validate_certs=params['validate_certs'])
    if response_code == 200:
        if response_data['percentComplete'] != -1:
            return (True, response_data['percentComplete'])
        else:
            return (False, response_data['percentComplete'])
    else:
        return (False, response_data)