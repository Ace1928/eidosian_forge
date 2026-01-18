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
def get_logging_status(self, headers=None):
    """
        Get the logging status for this bucket.

        :rtype: :class:`boto.s3.bucketlogging.BucketLogging`
        :return: A BucketLogging object for this bucket.
        """
    response = self.connection.make_request('GET', self.name, query_args='logging', headers=headers)
    body = response.read()
    if response.status == 200:
        blogging = BucketLogging()
        h = handler.XmlHandler(blogging, self)
        if not isinstance(body, bytes):
            body = body.encode('utf-8')
        xml.sax.parseString(body, h)
        return blogging
    else:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)