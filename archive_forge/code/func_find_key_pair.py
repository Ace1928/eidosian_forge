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
def find_key_pair(ec2_client, name):
    try:
        key = ec2_client.describe_key_pairs(aws_retry=True, KeyNames=[name])
    except is_boto3_error_code('InvalidKeyPair.NotFound'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as err:
        raise Ec2KeyFailure(err, 'error finding keypair')
    except IndexError:
        key = None
    return key['KeyPairs'][0]