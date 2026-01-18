import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def start_key_deletion(connection, module, key_metadata):
    if key_metadata['KeyState'] == 'PendingDeletion':
        return False
    if module.check_mode:
        return True
    deletion_params = {'KeyId': key_metadata['Arn']}
    if module.params.get('pending_window'):
        deletion_params['PendingWindowInDays'] = module.params.get('pending_window')
    try:
        connection.schedule_key_deletion(**deletion_params)
        return True
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to schedule key for deletion')