import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def build_unique_id(module):
    data = dict(module.params)
    [data.pop(each, None) for each in ('objects', 'timeout')]
    json_data = json.dumps(data, sort_keys=True).encode('utf-8')
    hashed_data = hashlib.md5(json_data).hexdigest()
    return hashed_data