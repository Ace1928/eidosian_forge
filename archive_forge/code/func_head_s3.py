import datetime
import fnmatch
import mimetypes
import os
import stat as osstat  # os.stat constants
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.etag import calculate_multipart_etag
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def head_s3(s3, bucket, s3keys):
    retkeys = []
    for entry in s3keys:
        retentry = entry.copy()
        try:
            retentry['s3_head'] = s3.head_object(Bucket=bucket, Key=entry['s3_path'])
        except is_boto3_error_code(['404', '403']):
            pass
        retkeys.append(retentry)
    return retkeys