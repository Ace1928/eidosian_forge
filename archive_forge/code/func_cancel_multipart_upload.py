from __future__ import division
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import BotoClientError
from boto.s3.acl import Policy, CannedACLStrings, Grant
from boto.s3.key import Key
from boto.s3.prefix import Prefix
from boto.s3.deletemarker import DeleteMarker
from boto.s3.multipart import MultiPartUpload
from boto.s3.multipart import CompleteMultiPartUpload
from boto.s3.multidelete import MultiDeleteResult
from boto.s3.multidelete import Error
from boto.s3.bucketlistresultset import BucketListResultSet
from boto.s3.bucketlistresultset import VersionedBucketListResultSet
from boto.s3.bucketlistresultset import MultiPartUploadListResultSet
from boto.s3.lifecycle import Lifecycle
from boto.s3.tagging import Tags
from boto.s3.cors import CORSConfiguration
from boto.s3.bucketlogging import BucketLogging
from boto.s3 import website
import boto.jsonresponse
import boto.utils
import xml.sax
import xml.sax.saxutils
import re
import base64
from collections import defaultdict
from boto.compat import BytesIO, six, StringIO, urllib
from boto.utils import get_utf8able_str
def cancel_multipart_upload(self, key_name, upload_id, headers=None):
    """
        To verify that all parts have been removed, so you don't get charged
        for the part storage, you should call the List Parts operation and
        ensure the parts list is empty.
        """
    query_args = 'uploadId=%s' % upload_id
    response = self.connection.make_request('DELETE', self.name, key_name, query_args=query_args, headers=headers)
    body = response.read()
    boto.log.debug(body)
    if response.status != 204:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)