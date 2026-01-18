from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ruleset_active(client, module, name):
    try:
        active_rule_set = client.describe_active_receipt_rule_set(aws_retry=True)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg="Couldn't get the active rule set.")
    if active_rule_set is not None and 'Metadata' in active_rule_set:
        return name == active_rule_set['Metadata']['Name']
    else:
        return False