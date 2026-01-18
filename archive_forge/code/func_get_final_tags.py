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
def get_final_tags(security_group_tags, specified_tags, purge_tags):
    if specified_tags is None:
        return security_group_tags
    tags_need_modify, tags_to_delete = compare_aws_tags(security_group_tags, specified_tags, purge_tags)
    end_result_tags = dict(((k, v) for k, v in specified_tags.items() if k not in tags_to_delete))
    end_result_tags.update(dict(((k, v) for k, v in security_group_tags.items() if k not in tags_to_delete)))
    end_result_tags.update(tags_need_modify)
    return end_result_tags