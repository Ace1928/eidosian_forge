from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def find_rule_id(module, system):
    """ Find the ID of the rule by name """
    rule_name = module.params['name']
    path = f'notifications/rules?name={rule_name}&fields=id'
    api_result = system.api.get(path=path)
    if len(api_result.get_json()['result']) > 0:
        result = api_result.get_json()['result'][0]
        rule_id = result['id']
    else:
        rule_id = None
    return rule_id