import base64
import re  # regex library
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.acm import ACMServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def pem_chain_split(module, pem):
    pem_arr = re.findall(pem_chain_split_regex, to_text(pem))
    if len(pem_arr) == 0:
        module.fail_json(msg='Unable to split certificate chain. Possibly zero-length chain?')
    return pem_arr