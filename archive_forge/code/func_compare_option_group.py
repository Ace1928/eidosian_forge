from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def compare_option_group(client, module):
    to_be_added = None
    to_be_removed = None
    current_option = get_option_group(client, module)
    new_options = module.params.get('options')
    new_settings = set([item['option_name'] for item in new_options])
    old_settings = set([item['option_name'] for item in current_option['options']])
    if new_settings != old_settings:
        to_be_added = list(new_settings - old_settings)
        to_be_removed = list(old_settings - new_settings)
    return (to_be_added, to_be_removed)