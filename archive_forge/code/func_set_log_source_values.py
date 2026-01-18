from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.ibm.qradar.plugins.module_utils.qradar import (
import json
def set_log_source_values(module, qradar_request):
    if module.params['type_name']:
        code, query_response = qradar_request.get('/api/config/event_sources/log_source_management/log_source_types?filter={0}'.format(quote('name="{0}"'.format(module.params['type_name']))))
        log_source_type_found = query_response[0]
    if module.params['type_id']:
        code, query_response = qradar_request.get('/api/config/event_sources/log_source_management/log_source_types?filter={0}'.format(quote('name="{0}"'.format(module.params['type_name']))))
        code, log_source_type_found = query_response[0]
    if log_source_type_found:
        if not module.params['type_id']:
            module.params['type_id'] = log_source_type_found['id']
    else:
        module.fail_json(msg='Incompatible type provided, please consult QRadar Documentation for Log Source Types')
    if module.params['protocol_type_id']:
        found_dict_in_list, _fdil_index = find_dict_in_list(log_source_type_found['protocol_types'], 'protocol_id', module.params['protocol_type_id'])
        if not found_dict_in_list:
            module.fail_json(msg='Incompatible protocol_type_id provided, please consult QRadar Documentation for Log Source Types')
    else:
        module.params['protocol_type_id'] = log_source_type_found['protocol_types'][0]['protocol_id']
    module.params['protocol_parameters'] = [{'id': module.params['protocol_type_id'], 'name': 'identifier', 'value': module.params['identifier']}]