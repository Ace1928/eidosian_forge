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
def merge_validation_into_config(config, validated_node, node_name):
    if validated_node is not None:
        if isinstance(validated_node, dict):
            config_node = config.get(node_name)
            if config_node is not None:
                config_node_items = list(config_node.items())
            else:
                config_node_items = []
            config[node_name] = dict(config_node_items + list(validated_node.items()))
        if isinstance(validated_node, list):
            config[node_name] = list(set(config.get(node_name) + validated_node))
    return config