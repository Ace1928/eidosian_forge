from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_load_balancer_attributes(connection, module, load_balancer_arn):
    try:
        attributes = connection.describe_load_balancer_attributes(aws_retry=True, LoadBalancerArn=load_balancer_arn)['Attributes']
        load_balancer_attributes = boto3_tag_list_to_ansible_dict(attributes)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe load balancer attributes')
    for k, v in list(load_balancer_attributes.items()):
        load_balancer_attributes[k.replace('.', '_')] = v
        del load_balancer_attributes[k]
    return load_balancer_attributes