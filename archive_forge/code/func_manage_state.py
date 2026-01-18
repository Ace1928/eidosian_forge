import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def manage_state(module, lambda_client):
    changed = False
    current_state = 'absent'
    state = module.params['state']
    action_taken = 'none'
    current_policy_statement = get_policy_statement(module, lambda_client)
    if current_policy_statement:
        current_state = 'present'
    if state == 'present':
        if current_state == 'present' and (not policy_equal(module, current_policy_statement)):
            remove_policy_permission(module, lambda_client)
            changed = add_policy_permission(module, lambda_client)
            action_taken = 'updated'
        if not current_state == 'present':
            changed = add_policy_permission(module, lambda_client)
            action_taken = 'added'
    elif current_state == 'present':
        changed = remove_policy_permission(module, lambda_client)
        action_taken = 'deleted'
    return dict(changed=changed, ansible_facts=dict(lambda_policy_action=action_taken))