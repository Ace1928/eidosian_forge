import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto3_conn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_aws_connection_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def set_api_sub_params(params):
    """
    Sets module sub-parameters to those expected by the boto3 API.

    :param params:
    :return:
    """
    api_params = dict()
    for param in params.keys():
        param_value = params.get(param, None)
        if param_value:
            api_params[pc(param)] = param_value
    return api_params