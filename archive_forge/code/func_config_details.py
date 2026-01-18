import json
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def config_details(client, module, function_name):
    """
    Returns configuration details for a lambda function.

    :param client: AWS API client reference (boto3)
    :param module: Ansible module reference
    :param function_name (str): Name of Lambda function to query
    :return dict:
    """
    lambda_info = dict()
    try:
        lambda_info.update(client.get_function_configuration(aws_retry=True, FunctionName=function_name))
    except is_boto3_error_code('ResourceNotFoundException'):
        lambda_info.update(function={})
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Trying to get {function_name} configuration')
    if 'Environment' in lambda_info and 'Variables' in lambda_info['Environment']:
        env_vars = lambda_info['Environment']['Variables']
        snaked_lambda_info = camel_dict_to_snake_dict(lambda_info)
        snaked_lambda_info['environment']['variables'] = env_vars
    else:
        snaked_lambda_info = camel_dict_to_snake_dict(lambda_info)
    return snaked_lambda_info