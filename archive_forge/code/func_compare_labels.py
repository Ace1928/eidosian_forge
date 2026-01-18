from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_labels(nodegroup_labels, param_labels):
    labels_to_unset = []
    labels_to_add_or_update = {}
    for label in nodegroup_labels.keys():
        if label not in param_labels:
            labels_to_unset.append(label)
    for key, value in param_labels.items():
        if key not in nodegroup_labels.keys():
            labels_to_add_or_update[key] = value
    return (labels_to_add_or_update, labels_to_unset)