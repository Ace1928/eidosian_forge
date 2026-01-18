import os
import uuid
from ansible.module_utils._text import to_bytes
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def get_key_fingerprint(check_mode, ec2_client, key_material):
    """
    EC2's fingerprints are non-trivial to generate, so push this key
    to a temporary name and make ec2 calculate the fingerprint for us.
    http://blog.jbrowne.com/?p=23
    https://forums.aws.amazon.com/thread.jspa?messageID=352828
    """
    name_in_use = True
    while name_in_use:
        random_name = 'ansible-' + str(uuid.uuid4())
        name_in_use = find_key_pair(ec2_client, random_name)
    temp_key = _import_key_pair(ec2_client, random_name, key_material)
    delete_key_pair(check_mode, ec2_client, random_name, finish_task=False)
    return temp_key['KeyFingerprint']