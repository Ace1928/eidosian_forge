import datetime
import functools
import time
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def compare_domain_versions(version1, version2):
    supported_engines = {'Elasticsearch': 1, 'OpenSearch': 2}
    if isinstance(version1, string_types):
        version1 = parse_version(version1)
    if isinstance(version2, string_types):
        version2 = parse_version(version2)
    if version1 is None and version2 is not None:
        return -1
    elif version1 is not None and version2 is None:
        return 1
    elif version1 is None and version2 is None:
        return 0
    e1 = supported_engines.get(version1.get('engine_type'))
    e2 = supported_engines.get(version2.get('engine_type'))
    if e1 < e2:
        return -1
    elif e1 > e2:
        return 1
    elif version1.get('major') < version2.get('major'):
        return -1
    elif version1.get('major') > version2.get('major'):
        return 1
    elif version1.get('minor') < version2.get('minor'):
        return -1
    elif version1.get('minor') > version2.get('minor'):
        return 1
    else:
        return 0