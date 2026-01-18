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
def s3_object_do_put(module, connection, connection_v4, s3_vars):
    if module.params.get('encryption_mode') == 'aws:kms':
        connection = connection_v4
    if s3_vars['src'] is not None and (not path_check(s3_vars['src'])):
        module.fail_json(msg=f'Local object "{s3_vars['src']}" does not exist for PUT operation')
    keyrtn = key_check(module, connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'], validate=s3_vars['validate'])
    bincontent = get_binary_content(s3_vars)
    if keyrtn and s3_vars['overwrite'] != 'always':
        if s3_vars['overwrite'] == 'never' or etag_compare(module, connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'], local_file=s3_vars['src'], content=bincontent):
            tags, tags_update = ensure_tags(connection, module, s3_vars['bucket'], s3_vars['object'])
            get_download_url(module, connection, s3_vars['bucket'], s3_vars['object'], s3_vars['expiry'], tags, changed=tags_update)
    if not s3_vars['acl_disabled']:
        s3_vars['permission'] = s3_vars['object_acl']
    upload_s3file(module, connection, s3_vars['bucket'], s3_vars['object'], s3_vars['expiry'], s3_vars['metadata'], s3_vars['encrypt'], s3_vars['headers'], src=s3_vars['src'], content=bincontent, acl_disabled=s3_vars['acl_disabled'])
    module.exit_json(failed=False)