from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def prop_to_dict(p):
    """convert properties to dictionary"""
    if len(p) == 0:
        return {}
    r_dict = {}
    for s in p.decode().split('\n'):
        kv = s.split('=')
        r_dict[kv[0].strip()] = kv[1].strip()
    return r_dict