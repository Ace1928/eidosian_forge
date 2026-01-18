import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_and_update_web_acl(client, module, web_acl_id):
    acl = get_web_acl(client, module, web_acl_id)
    rule_lookup = create_rule_lookup(client, module)
    existing_rules = acl['Rules']
    desired_rules = [{'RuleId': rule_lookup[rule['name']]['RuleId'], 'Priority': rule['priority'], 'Action': {'Type': rule['action'].upper()}, 'Type': rule.get('type', 'regular').upper()} for rule in module.params['rules']]
    missing = [rule for rule in desired_rules if rule not in existing_rules]
    extras = []
    if module.params['purge_rules']:
        extras = [rule for rule in existing_rules if rule not in desired_rules]
    insertions = [format_for_update(rule, 'INSERT') for rule in missing]
    deletions = [format_for_update(rule, 'DELETE') for rule in extras]
    changed = bool(insertions + deletions)
    params = {'WebACLId': acl['WebACLId'], 'DefaultAction': acl['DefaultAction']}
    change_tokens = []
    if deletions:
        try:
            params['Updates'] = deletions
            result = run_func_with_change_token_backoff(client, module, params, client.update_web_acl)
            change_tokens.append(result['ChangeToken'])
            get_waiter(client, 'change_token_in_sync').wait(ChangeToken=result['ChangeToken'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not update Web ACL')
    if insertions:
        try:
            params['Updates'] = insertions
            result = run_func_with_change_token_backoff(client, module, params, client.update_web_acl)
            change_tokens.append(result['ChangeToken'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not update Web ACL')
    if change_tokens:
        for token in change_tokens:
            get_waiter(client, 'change_token_in_sync').wait(ChangeToken=token)
    if changed:
        acl = get_web_acl(client, module, web_acl_id)
    return (changed, acl)