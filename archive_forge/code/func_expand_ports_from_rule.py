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
def expand_ports_from_rule(rule):
    if rule.get('icmp_type') is not None:
        return ({'from_port': rule.get('icmp_type'), 'to_port': rule.get('icmp_code')},)
    if rule.get('from_port') is not None or rule.get('to_port') is not None:
        return ({'from_port': rule.get('from_port'), 'to_port': rule.get('to_port')},)
    if rule.get('ports') is not None:
        ports = expand_ports_list(rule.get('ports'))
        return tuple(({'from_port': from_port, 'to_port': to_port} for from_port, to_port in ports))
    return ({},)