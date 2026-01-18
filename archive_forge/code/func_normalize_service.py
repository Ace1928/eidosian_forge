from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def normalize_service(service):
    normalized = camel_dict_to_snake_dict(service, ignore_list=['Tags'])
    normalized['tags'] = boto3_tag_list_to_ansible_dict(service.get('Tags'))
    return normalized