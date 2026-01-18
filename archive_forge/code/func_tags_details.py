import json
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def tags_details(client, module, function_name):
    """
    Returns tag details for a lambda function.

    :param client: AWS API client reference (boto3)
    :param module: Ansible module reference
    :param function_name (str): Name of Lambda function to query
    :return dict:
    """
    lambda_info = dict()
    try:
        lambda_info.update(tags=client.get_function(aws_retry=True, FunctionName=function_name).get('Tags', {}))
    except is_boto3_error_code('ResourceNotFoundException'):
        lambda_info.update(function={})
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Trying to get {function_name} tags')
    return camel_dict_to_snake_dict(lambda_info)