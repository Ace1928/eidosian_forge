import itertools
import json
import re
from collections import namedtuple
from copy import deepcopy
from ipaddress import ip_network
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_ipv6_subnet
from ansible.module_utils.common.network import to_subnet
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_id
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def flatten_nested_targets(module, rules):

    def _flatten(targets):
        for target in targets:
            if isinstance(target, list):
                module.deprecate('Support for nested lists in cidr_ip and cidr_ipv6 has been deprecated.  The flatten filter can be used instead.', date='2024-12-01', collection_name='amazon.aws')
                for t in _flatten(target):
                    yield t
            elif isinstance(target, string_types):
                yield target
    if rules is not None:
        for rule in rules:
            target_list_type = None
            if isinstance(rule.get('cidr_ip'), list):
                target_list_type = 'cidr_ip'
            elif isinstance(rule.get('cidr_ipv6'), list):
                target_list_type = 'cidr_ipv6'
            if target_list_type is not None:
                rule[target_list_type] = list(_flatten(rule[target_list_type]))
    return rules