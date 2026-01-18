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
def s3_object_do_copy(module, connection, connection_v4, s3_vars):
    copy_src = module.params.get('copy_src')
    if not copy_src.get('object') and s3_vars['object']:
        module.fail_json(msg='A destination object was specified while trying to copy all the objects from the source bucket.')
    src_bucket = copy_src.get('bucket')
    if not copy_src.get('object'):
        keys = list_keys(connection, src_bucket, copy_src.get('prefix'))
        if len(keys) == 0:
            module.exit_json(msg=f'No object found to be copied from source bucket {src_bucket}.')
        changed = False
        number_keys_updated = 0
        for key in keys:
            updated, result = copy_object_to_bucket(module, connection, s3_vars['bucket'], key, s3_vars['encrypt'], s3_vars['metadata'], s3_vars['validate'], src_bucket, key, versionId=copy_src.get('version_id'))
            changed |= updated
            number_keys_updated += 1 if updated else 0
        msg = f"object(s) from buckets '{src_bucket}' and '{s3_vars['bucket']}' are the same."
        if number_keys_updated:
            msg = f"{number_keys_updated} copied into bucket '{s3_vars['bucket']}'"
        module.exit_json(changed=changed, msg=msg)
    else:
        changed, result = copy_object_to_bucket(module, connection, s3_vars['bucket'], s3_vars['object'], s3_vars['encrypt'], s3_vars['metadata'], s3_vars['validate'], src_bucket, copy_src.get('object'), versionId=copy_src.get('version_id'))
        module.exit_json(changed=changed, **result)