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
def list_multipart_uploads(self, key_marker='', upload_id_marker='', headers=None, encoding_type=None):
    """
        List multipart upload objects within a bucket.  This returns an
        instance of an MultiPartUploadListResultSet that automatically
        handles all of the result paging, etc. from S3.  You just need
        to keep iterating until there are no more results.

        :type key_marker: string
        :param key_marker: The "marker" of where you are in the result set

        :type upload_id_marker: string
        :param upload_id_marker: The upload identifier

        :param encoding_type: Requests Amazon S3 to encode the response and
            specifies the encoding method to use.

            An object key can contain any Unicode character; however, XML 1.0
            parser cannot parse some characters, such as characters with an
            ASCII value from 0 to 10. For characters that are not supported in
            XML 1.0, you can add this parameter to request that Amazon S3
            encode the keys in the response.

            Valid options: ``url``
        :type encoding_type: string

        :rtype: :class:`boto.s3.bucketlistresultset.MultiPartUploadListResultSet`
        :return: an instance of a BucketListResultSet that handles paging, etc
        """
    return MultiPartUploadListResultSet(self, key_marker, upload_id_marker, headers, encoding_type=encoding_type)