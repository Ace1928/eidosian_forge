from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_taints(nodegroup_taints, param_taints):
    taints_to_unset = []
    taints_to_add_or_update = []
    for taint in nodegroup_taints:
        if taint not in param_taints:
            taints_to_unset.append(taint)
    for taint in param_taints:
        if taint not in nodegroup_taints:
            taints_to_add_or_update.append(taint)
    return (taints_to_add_or_update, taints_to_unset)