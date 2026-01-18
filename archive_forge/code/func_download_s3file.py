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
def download_s3file(module, s3, bucket, obj, dest, retries, version=None):
    if module.check_mode:
        module.exit_json(msg='GET operation skipped - running in check mode', changed=True)
    try:
        if version:
            s3.get_object(aws_retry=True, Bucket=bucket, Key=obj, VersionId=version)
        else:
            s3.get_object(aws_retry=True, Bucket=bucket, Key=obj)
    except is_boto3_error_code(['404', '403']) as e:
        module.fail_json_aws(e, msg=f'Could not find the key {obj}.')
    except is_boto3_error_message('require AWS Signature Version 4'):
        raise Sigv4Required()
    except is_boto3_error_code('InvalidArgument') as e:
        module.fail_json_aws(e, msg=f'Could not find the key {obj}.')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError, boto3.exceptions.Boto3Error) as e:
        raise S3ObjectFailure(f'Could not find the key {obj}.', e)
    optional_kwargs = {'ExtraArgs': {'VersionId': version}} if version else {}
    for x in range(0, retries + 1):
        try:
            s3.download_file(bucket, obj, dest, aws_retry=True, **optional_kwargs)
            module.exit_json(msg='GET operation complete', changed=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError, boto3.exceptions.Boto3Error) as e:
            if x >= retries:
                raise S3ObjectFailure(f'Failed while downloading {obj}.', e)
        except SSLError as e:
            if x >= retries:
                module.fail_json_aws(e, msg='s3 download failed')