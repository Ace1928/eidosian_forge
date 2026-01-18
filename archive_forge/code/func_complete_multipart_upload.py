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
def complete_multipart_upload(self, key_name, upload_id, xml_body, headers=None):
    """
        Complete a multipart upload operation.
        """
    query_args = 'uploadId=%s' % upload_id
    if headers is None:
        headers = {}
    headers['Content-Type'] = 'text/xml'
    response = self.connection.make_request('POST', self.name, key_name, query_args=query_args, headers=headers, data=xml_body)
    contains_error = False
    body = response.read().decode('utf-8')
    if body.find('<Error>') > 0:
        contains_error = True
    boto.log.debug(body)
    if response.status == 200 and (not contains_error):
        resp = CompleteMultiPartUpload(self)
        h = handler.XmlHandler(resp, self)
        if not isinstance(body, bytes):
            body = body.encode('utf-8')
        xml.sax.parseString(body, h)
        k = self.key_class(self)
        k.handle_version_headers(response)
        k.handle_encryption_headers(response)
        resp.version_id = k.version_id
        resp.encrypted = k.encrypted
        return resp
    else:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)