from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def describe_log_group(client, log_group_name, module):
    params = {}
    if log_group_name:
        params['logGroupNamePrefix'] = log_group_name
    try:
        paginator = client.get_paginator('describe_log_groups')
        desc_log_group = paginator.paginate(**params).build_full_result()
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Unable to describe log group {log_group_name}')
    for log_group in desc_log_group['logGroups']:
        log_group_name = log_group['logGroupName']
        try:
            tags = client.list_tags_log_group(logGroupName=log_group_name)
        except is_boto3_error_code('AccessDeniedException'):
            tags = {}
            module.warn(f'Permission denied listing tags for log group {log_group_name}')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f'Unable to describe tags for log group {log_group_name}')
        log_group['tags'] = tags.get('tags', {})
    return desc_log_group