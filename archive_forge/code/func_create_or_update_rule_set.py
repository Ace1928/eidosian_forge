from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_rule_set(client, module):
    name = module.params.get('name')
    check_mode = module.check_mode
    changed = False
    rule_sets = list_rule_sets(client, module)
    if not rule_set_in(name, rule_sets):
        if not check_mode:
            try:
                client.create_receipt_rule_set(RuleSetName=name, aws_retry=True)
            except (BotoCoreError, ClientError) as e:
                module.fail_json_aws(e, msg=f"Couldn't create rule set {name}.")
        changed = True
        rule_sets = list(rule_sets)
        rule_sets.append({'Name': name})
    active_changed, active = update_active_rule_set(client, module, name, module.params.get('active'))
    changed |= active_changed
    module.exit_json(changed=changed, active=active, rule_sets=[camel_dict_to_snake_dict(x) for x in rule_sets])