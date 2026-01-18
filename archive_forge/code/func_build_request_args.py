from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def build_request_args(eni_id, filters):
    request_args = {'NetworkInterfaceIds': [eni_id] if eni_id else [], 'Filters': ansible_dict_to_boto3_filter_list(filters)}
    request_args = {k: v for k, v in request_args.items() if v}
    return request_args