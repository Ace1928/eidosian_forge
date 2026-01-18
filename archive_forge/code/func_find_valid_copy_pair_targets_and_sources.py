from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request
def find_valid_copy_pair_targets_and_sources(params):
    get_status = 'storage-systems/%s/volumes' % params['ssid']
    url = params['api_url'] + get_status
    response_code, response_data = request(url, ignore_errors=True, method='GET', url_username=params['api_username'], url_password=params['api_password'], headers=HEADERS, validate_certs=params['validate_certs'])
    if response_code == 200:
        source_capacity = None
        candidates = []
        for volume in response_data:
            if volume['id'] == params['search_volume_id']:
                source_capacity = volume['capacity']
            else:
                candidates.append(volume)
        potential_sources = []
        potential_targets = []
        for volume in candidates:
            if volume['capacity'] > source_capacity:
                if volume['volumeCopyTarget'] is False:
                    if volume['volumeCopySource'] is False:
                        potential_targets.append(volume['id'])
            elif volume['volumeCopyTarget'] is False:
                if volume['volumeCopySource'] is False:
                    potential_sources.append(volume['id'])
        return (potential_targets, potential_sources)
    else:
        raise Exception('Response [%s]' % response_code)