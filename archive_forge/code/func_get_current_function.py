import base64
import hashlib
import re
import traceback
from collections import Counter
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_current_function(connection, function_name, qualifier=None):
    try:
        if qualifier is not None:
            return connection.get_function(FunctionName=function_name, Qualifier=qualifier, aws_retry=True)
        return connection.get_function(FunctionName=function_name, aws_retry=True)
    except is_boto3_error_code('ResourceNotFoundException'):
        return None