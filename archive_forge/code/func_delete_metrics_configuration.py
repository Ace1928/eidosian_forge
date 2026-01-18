from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_metrics_configuration(client, module):
    bucket_name = module.params.get('bucket_name')
    mc_id = module.params.get('id')
    try:
        client.get_bucket_metrics_configuration(aws_retry=True, Bucket=bucket_name, Id=mc_id)
    except is_boto3_error_code('NoSuchConfiguration'):
        module.exit_json(changed=False)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Failed to get bucket metrics configuration')
    if module.check_mode:
        module.exit_json(changed=True)
    try:
        client.delete_bucket_metrics_configuration(aws_retry=True, Bucket=bucket_name, Id=mc_id)
    except is_boto3_error_code('NoSuchConfiguration'):
        module.exit_json(changed=False)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f"Failed to delete bucket metrics configuration '{mc_id}'")
    module.exit_json(changed=True)