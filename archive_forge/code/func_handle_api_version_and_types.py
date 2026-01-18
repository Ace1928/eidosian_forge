from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def handle_api_version_and_types(module, payload):
    payload_count = 0
    for workflow in module.params['workflows']:
        handle_handler_api_and_type(payload['workflows'][payload_count]['handler'])
        if workflow.get('mutator'):
            handle_mutator_api_and_type(payload['workflows'][payload_count]['mutator'])
        elif 'mutator' in payload['workflows'][payload_count]:
            payload['workflows'][payload_count].pop('mutator')
        if workflow.get('filters'):
            handle_filter_api_and_type(payload['workflows'][payload_count]['filters'], workflow)
        elif 'filters' in payload['workflows'][payload_count]:
            payload['workflows'][payload_count].pop('filters')
        payload_count += 1