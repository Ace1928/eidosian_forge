import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_key_details(connection, module, key_id):
    try:
        result = get_kms_metadata_with_backoff(connection, key_id)['KeyMetadata']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to obtain key metadata')
    result['KeyArn'] = result.pop('Arn')
    try:
        aliases = get_kms_aliases_lookup(connection)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to obtain aliases')
    try:
        current_rotation_status = connection.get_key_rotation_status(KeyId=key_id)
        result['enable_key_rotation'] = current_rotation_status.get('KeyRotationEnabled')
    except is_boto3_error_code(['AccessDeniedException', 'UnsupportedOperationException']) as e:
        result['enable_key_rotation'] = None
    result['aliases'] = aliases.get(result['KeyId'], [])
    result = camel_dict_to_snake_dict(result)
    try:
        result['grants'] = [camel_to_snake_grant(grant) for grant in get_kms_grants_with_backoff(connection, key_id)['Grants']]
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to obtain key grants')
    tags = get_kms_tags(connection, module, key_id)
    result['tags'] = boto3_tag_list_to_ansible_dict(tags, 'TagKey', 'TagValue')
    result['policies'] = get_kms_policies(connection, module, key_id)
    result['key_policies'] = [json.loads(policy) for policy in result['policies']]
    return result