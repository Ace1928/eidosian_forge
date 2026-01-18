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
def s3_object_do_get(module, connection, connection_v4, s3_vars):
    if module.params.get('sig_v4'):
        connection = connection_v4
    keyrtn = key_check(module, connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'], validate=s3_vars['validate'])
    if not keyrtn:
        if s3_vars['version']:
            module.fail_json(msg=f'Key {s3_vars['object']} with version id {s3_vars['version']} does not exist.')
        module.fail_json(msg=f'Key {s3_vars['object']} does not exist.')
    if s3_vars['dest'] and path_check(s3_vars['dest']) and (s3_vars['overwrite'] != 'always'):
        if s3_vars['overwrite'] == 'never':
            module.exit_json(msg='Local object already exists and overwrite is disabled.', changed=False)
        if s3_vars['overwrite'] == 'different' and etag_compare(module, connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'], local_file=s3_vars['dest']):
            module.exit_json(msg='Local and remote object are identical, ignoring. Use overwrite=always parameter to force.', changed=False)
        if s3_vars['overwrite'] == 'latest' and is_local_object_latest(connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'], local_file=s3_vars['dest']):
            module.exit_json(msg='Local object is latest, ignoreing. Use overwrite=always parameter to force.', changed=False)
    try:
        download_s3file(module, connection, s3_vars['bucket'], s3_vars['object'], s3_vars['dest'], s3_vars['retries'], version=s3_vars['version'])
    except Sigv4Required:
        download_s3file(module, connection_v4, s3_vars['bucket'], s3_vars['obj'], s3_vars['dest'], s3_vars['retries'], version=s3_vars['version'])
    module.exit_json(failed=False)