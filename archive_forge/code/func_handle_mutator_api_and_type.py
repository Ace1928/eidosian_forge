from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def handle_mutator_api_and_type(payload_mutator):
    payload_mutator['type'] = MUTATOR_TYPE[payload_mutator['type']]
    payload_mutator['api_version'] = API_VERSION['v2']