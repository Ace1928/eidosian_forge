import base64
import re  # regex library
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.acm import ACMServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def chain_compare(module, a, b):
    chain_a_pem = pem_chain_split(module, a)
    chain_b_pem = pem_chain_split(module, b)
    if len(chain_a_pem) != len(chain_b_pem):
        return False
    for ca, cb in zip(chain_a_pem, chain_b_pem):
        der_a = PEM_body_to_DER(module, ca)
        der_b = PEM_body_to_DER(module, cb)
        if der_a != der_b:
            return False
    return True