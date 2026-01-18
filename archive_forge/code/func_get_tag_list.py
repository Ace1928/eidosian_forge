from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_tag_list(keys, tags):
    """
    Returns a list of dicts with tags to act on
    keys : set of keys to get the values for
    tags : the dict of tags to turn into a list
    """
    tag_list = []
    for k in keys:
        tag_list.append({'Key': k, 'Value': tags[k]})
    return tag_list