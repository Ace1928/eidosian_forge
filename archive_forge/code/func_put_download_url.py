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
def put_download_url(s3, bucket, obj, expiry):
    try:
        url = s3.generate_presigned_url(ClientMethod='put_object', Params={'Bucket': bucket, 'Key': obj}, ExpiresIn=expiry)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError, boto3.exceptions.Boto3Error) as e:
        raise S3ObjectFailure('Unable to generate presigned URL', e)
    return url