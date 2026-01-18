import datetime
import itertools
import time
import uuid
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_operation_preferences(module):
    params = dict()
    if module.params.get('regions'):
        params['RegionOrder'] = list(module.params['regions'])
    for param, api_name in {'fail_count': 'FailureToleranceCount', 'fail_percentage': 'FailureTolerancePercentage', 'parallel_percentage': 'MaxConcurrentPercentage', 'parallel_count': 'MaxConcurrentCount'}.items():
        if module.params.get('failure_tolerance', {}).get(param):
            params[api_name] = module.params.get('failure_tolerance', {}).get(param)
    return params