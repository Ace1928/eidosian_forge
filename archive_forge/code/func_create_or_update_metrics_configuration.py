from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_metrics_configuration(client, module):
    bucket_name = module.params.get('bucket_name')
    mc_id = module.params.get('id')
    filter_prefix = module.params.get('filter_prefix')
    filter_tags = module.params.get('filter_tags')
    try:
        response = client.get_bucket_metrics_configuration(aws_retry=True, Bucket=bucket_name, Id=mc_id)
        metrics_configuration = response['MetricsConfiguration']
    except is_boto3_error_code('NoSuchConfiguration'):
        metrics_configuration = None
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Failed to get bucket metrics configuration')
    new_configuration = _create_metrics_configuration(mc_id, filter_prefix, filter_tags)
    if metrics_configuration:
        if metrics_configuration == new_configuration:
            module.exit_json(changed=False)
    if module.check_mode:
        module.exit_json(changed=True)
    try:
        client.put_bucket_metrics_configuration(aws_retry=True, Bucket=bucket_name, Id=mc_id, MetricsConfiguration=new_configuration)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f"Failed to put bucket metrics configuration '{mc_id}'")
    module.exit_json(changed=True)