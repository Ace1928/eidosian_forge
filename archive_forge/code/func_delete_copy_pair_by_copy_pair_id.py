from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request
def delete_copy_pair_by_copy_pair_id(params):
    get_status = 'storage-systems/%s/volume-copy-jobs/%s?retainRepositories=false' % (params['ssid'], params['volume_copy_pair_id'])
    url = params['api_url'] + get_status
    rc, resp = request(url, ignore_errors=True, method='DELETE', url_username=params['api_username'], url_password=params['api_password'], headers=HEADERS, validate_certs=params['validate_certs'])
    if rc != 204:
        return (False, (rc, resp))
    else:
        return (True, (rc, resp))