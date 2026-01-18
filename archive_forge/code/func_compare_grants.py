import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def compare_grants(existing_grants, desired_grants, purge_grants=False):
    existing_dict = dict(((eg['name'], eg) for eg in existing_grants))
    desired_dict = dict(((dg['name'], dg) for dg in desired_grants))
    to_add_keys = set(desired_dict.keys()) - set(existing_dict.keys())
    if purge_grants:
        to_remove_keys = set(existing_dict.keys()) - set(desired_dict.keys())
    else:
        to_remove_keys = set()
    to_change_candidates = set(existing_dict.keys()) & set(desired_dict.keys())
    for candidate in to_change_candidates:
        if different_grant(existing_dict[candidate], desired_dict[candidate]):
            to_add_keys.add(candidate)
            to_remove_keys.add(candidate)
    to_add = []
    to_remove = []
    for key in to_add_keys:
        grant = desired_dict[key]
        to_add.append(grant)
    for key in to_remove_keys:
        grant = existing_dict[key]
        to_remove.append(grant)
    return (to_add, to_remove)