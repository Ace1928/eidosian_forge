from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_rule_set(client, module):
    name = module.params.get('name')
    check_mode = module.check_mode
    changed = False
    rule_sets = list_rule_sets(client, module)
    if rule_set_in(name, rule_sets):
        active = ruleset_active(client, module, name)
        if active and (not module.params.get('force')):
            module.fail_json(msg=f"Couldn't delete rule set {name} because it is currently active. Set force=true to delete an active ruleset.", error={'code': 'CannotDelete', 'message': f'Cannot delete active rule set: {name}'})
        if not check_mode:
            if active and module.params.get('force'):
                deactivate_rule_set(client, module)
            try:
                client.delete_receipt_rule_set(RuleSetName=name, aws_retry=True)
            except (BotoCoreError, ClientError) as e:
                module.fail_json_aws(e, msg=f"Couldn't delete rule set {name}.")
        changed = True
        rule_sets = [x for x in rule_sets if x['Name'] != name]
    module.exit_json(changed=changed, rule_sets=[camel_dict_to_snake_dict(x) for x in rule_sets])