import base64
import hashlib
import re
import traceback
from collections import Counter
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def sha256sum(filename):
    hasher = hashlib.sha256()
    with open(filename, 'rb') as f:
        hasher.update(f.read())
    code_hash = hasher.digest()
    code_b64 = base64.b64encode(code_hash)
    hex_digest = code_b64.decode('utf-8')
    return hex_digest