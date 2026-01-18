import datetime
import re
from collections import OrderedDict
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def add_key_else_validate(self, dict_object, key_name, attribute_name, value_to_set, valid_values, to_aws_list=False):
    if key_name in dict_object:
        self.validate_attribute_with_allowed_values(value_to_set, attribute_name, valid_values)
    elif to_aws_list:
        dict_object[key_name] = ansible_list_to_cloudfront_list(value_to_set)
    elif value_to_set is not None:
        dict_object[key_name] = value_to_set
    return dict_object