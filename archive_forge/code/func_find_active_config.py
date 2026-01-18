from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_active_config(client, module):
    """
    looking for configuration by name
    """
    name = module.params['name']
    try:
        all_configs = get_configurations_with_backoff(client)['Configurations']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='failed to obtain kafka configurations')
    active_configs = list((item for item in all_configs if item['Name'] == name and item['State'] == 'ACTIVE'))
    if active_configs:
        if len(active_configs) == 1:
            return active_configs[0]
        else:
            module.fail_json_aws(msg=f"found more than one active config with name '{name}'")
    return None