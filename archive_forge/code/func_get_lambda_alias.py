import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_lambda_alias(module_params, client):
    """
    Returns the lambda function alias if it exists.

    :param module_params: AnsibleAWSModule parameters
    :param client: (wrapped) boto3 lambda client
    :return:
    """
    api_params = set_api_params(module_params, ('function_name', 'name'))
    try:
        results = client.get_alias(aws_retry=True, **api_params)
    except is_boto3_error_code('ResourceNotFoundException'):
        results = None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise LambdaAnsibleAWSError('Error retrieving function alias', exception=e)
    return results