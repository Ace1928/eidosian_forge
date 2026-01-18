from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_instance_tags(conn, arn):
    try:
        return boto3_tag_list_to_ansible_dict(conn.list_tags_for_resource(ResourceName=arn, aws_retry=True)['TagList'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise RdsInstanceInfoFailure(e, f"Couldn't get tags for instance {arn}")