from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.community.general.plugins.module_utils.scaleway import (
from ansible.module_utils.basic import AnsibleModule
def payload_from_wished_cn(wished_cn):
    payload = {'namespace_id': wished_cn['namespace_id'], 'name': wished_cn['name'], 'description': wished_cn['description'], 'min_scale': wished_cn['min_scale'], 'max_scale': wished_cn['max_scale'], 'environment_variables': wished_cn['environment_variables'], 'secret_environment_variables': SecretVariables.dict_to_list(wished_cn['secret_environment_variables']), 'memory_limit': wished_cn['memory_limit'], 'timeout': wished_cn['timeout'], 'privacy': wished_cn['privacy'], 'registry_image': wished_cn['registry_image'], 'max_concurrency': wished_cn['max_concurrency'], 'protocol': wished_cn['protocol'], 'port': wished_cn['port'], 'redeploy': wished_cn['redeploy']}
    return payload