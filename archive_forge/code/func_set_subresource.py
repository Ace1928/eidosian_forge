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
def set_subresource(self, subresource, value, key_name='', headers=None, version_id=None):
    """
        Set a subresource for a bucket or key.

        :type subresource: string
        :param subresource: The subresource to set.

        :type value: string
        :param value: The value of the subresource.

        :type key_name: string
        :param key_name: The key to operate on, or None to operate on the
            bucket.

        :type headers: dict
        :param headers: Additional HTTP headers to include in the request.

        :type src_version_id: string
        :param src_version_id: Optional. The version id of the key to
            operate on. If not specified, operate on the newest
            version.
        """
    if not subresource:
        raise TypeError('set_subresource called with subresource=None')
    query_args = subresource
    if version_id:
        query_args += '&versionId=%s' % version_id
    if not isinstance(value, bytes):
        value = value.encode('utf-8')
    response = self.connection.make_request('PUT', self.name, key_name, data=value, query_args=query_args, headers=headers)
    body = response.read()
    if response.status != 200:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)