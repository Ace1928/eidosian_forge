from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_load_balancer_tags(connection, module, load_balancer_arn):
    try:
        tag_descriptions = connection.describe_tags(aws_retry=True, ResourceArns=[load_balancer_arn])['TagDescriptions']
        return boto3_tag_list_to_ansible_dict(tag_descriptions[0]['Tags'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe load balancer tags')