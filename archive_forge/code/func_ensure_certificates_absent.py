import base64
import re  # regex library
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.acm import ACMServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_certificates_absent(client, module, acm, certificates):
    for cert in certificates:
        if not module.check_mode:
            acm.delete_certificate(client, module, cert['certificate_arn'])
    module.exit_json(arns=[cert['certificate_arn'] for cert in certificates], changed=len(certificates) > 0)