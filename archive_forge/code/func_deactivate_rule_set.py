from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def deactivate_rule_set(client, module):
    try:
        client.set_active_receipt_rule_set(aws_retry=True)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg="Couldn't set active rule set to None.")