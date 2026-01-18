import base64
import copy
import io
import mimetypes
import os
import time
from ssl import SSLError
from ansible.module_utils.basic import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import HAS_MD5
from ansible_collections.amazon.aws.plugins.module_utils.s3 import calculate_etag
from ansible_collections.amazon.aws.plugins.module_utils.s3 import calculate_etag_content
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def populate_params(module):
    variable_dict = copy.deepcopy(module.params)
    if variable_dict['validate_bucket_name']:
        validate_bucket_name(variable_dict['bucket'])
    if variable_dict.get('overwrite') == 'different' and (not HAS_MD5):
        module.fail_json(msg='overwrite=different is unavailable: ETag calculation requires MD5 support')
    if variable_dict.get('overwrite') not in ['always', 'never', 'different', 'latest']:
        if module.boolean(variable_dict['overwrite']):
            variable_dict['overwrite'] = 'always'
        else:
            variable_dict['overwrite'] = 'never'
    if variable_dict['object']:
        if variable_dict.get('mode') == 'delete':
            module.fail_json(msg='Parameter object cannot be used with mode=delete')
        obj = variable_dict['object']
        if obj.startswith('/'):
            obj = obj[1:]
            variable_dict['object'] = obj
            module.deprecate("Support for passing object key names with a leading '/' has been deprecated.", date='2025-12-01', collection_name='amazon.aws')
    variable_dict['validate'] = not variable_dict['ignore_nonexistent_bucket']
    variable_dict['acl_disabled'] = False
    return variable_dict