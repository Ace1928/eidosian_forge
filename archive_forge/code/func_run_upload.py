import io
import logging
import math
import re
import urllib
import eventlet
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _
import glance_store.location
def run_upload(s3_client, bucket, key, part):
    """Upload the upload part into S3 and set returned etag and size to its
    part info.

    :param s3_client: An object with credentials to connect to S3
    :param bucket: The S3 bucket name
    :param key: The object name to be stored (image identifier)
    :param part: UploadPart object which used during multipart upload
    """
    pnum = part.partnum
    bsize = part.chunks
    upload_id = part.mpu['UploadId']
    LOG.debug('Uploading upload part in S3 partnum=%(pnum)d, size=%(bsize)d, key=%(key)s, UploadId=%(UploadId)s', {'pnum': pnum, 'bsize': bsize, 'key': key, 'UploadId': upload_id})
    try:
        key = s3_client.upload_part(Body=part.fp, Bucket=bucket, ContentLength=bsize, Key=key, PartNumber=pnum, UploadId=upload_id)
        part.etag[part.partnum] = key['ETag']
        part.size = bsize
    except boto_exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        LOG.warning('Failed to upload part in S3 partnum=%(pnum)d, size=%(bsize)d, error code=%(error_code)d, error message=%(error_message)s', {'pnum': pnum, 'bsize': bsize, 'error_code': error_code, 'error_message': error_message})
        part.success = False
    finally:
        part.fp.close()