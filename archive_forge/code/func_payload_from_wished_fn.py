from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.community.general.plugins.module_utils.scaleway import (
from ansible.module_utils.basic import AnsibleModule
def payload_from_wished_fn(wished_fn):
    payload = {'project_id': wished_fn['project_id'], 'name': wished_fn['name'], 'description': wished_fn['description'], 'environment_variables': wished_fn['environment_variables'], 'secret_environment_variables': SecretVariables.dict_to_list(wished_fn['secret_environment_variables'])}
    return payload