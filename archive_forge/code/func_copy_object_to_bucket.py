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
def copy_object_to_bucket(module, s3, bucket, obj, encrypt, metadata, validate, src_bucket, src_obj, versionId=None):
    try:
        params = {'Bucket': bucket, 'Key': obj}
        if not key_check(module, s3, src_bucket, src_obj, version=versionId, validate=validate):
            module.exit_json(msg=f'Key {src_obj} does not exist in bucket {src_bucket}.', changed=False)
        s_etag = get_etag(s3, src_bucket, src_obj, version=versionId)
        d_etag = get_etag(s3, bucket, obj)
        if s_etag == d_etag:
            if module.check_mode:
                changed = check_object_tags(module, s3, bucket, obj)
                result = {}
                if changed:
                    result.update({'msg': 'Would have update object tags is not running in check mode.'})
                return (changed, result)
            tags, changed = ensure_tags(s3, module, bucket, obj)
            result = {'msg': 'ETag from source and destination are the same'}
            if changed:
                result = {'msg': 'tags successfully updated.', 'tags': tags}
            return (changed, result)
        elif module.check_mode:
            return (True, {'msg': 'ETag from source and destination differ'})
        else:
            changed = True
            bucketsrc = {'Bucket': src_bucket, 'Key': src_obj}
            if versionId:
                bucketsrc.update({'VersionId': versionId})
            params.update({'CopySource': bucketsrc})
            params.update(get_extra_params(encrypt, module.params.get('encryption_mode'), module.params.get('encryption_kms_key_id'), metadata))
            s3.copy_object(aws_retry=True, **params)
            put_object_acl(module, s3, bucket, obj)
            tags, tags_updated = ensure_tags(s3, module, bucket, obj)
            msg = f'Object copied from bucket {bucketsrc['Bucket']} to bucket {bucket}.'
            return (changed, {'msg': msg, 'tags': tags})
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError, boto3.exceptions.Boto3Error) as e:
        raise S3ObjectFailure(f'Failed while copying object {obj} from bucket {module.params['copy_src'].get('Bucket')}.', e)