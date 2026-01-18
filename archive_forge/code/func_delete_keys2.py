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
def delete_keys2(hdrs):
    hdrs = hdrs or {}
    data = u'<?xml version="1.0" encoding="UTF-8"?>'
    data += u'<Delete>'
    if quiet:
        data += u'<Quiet>true</Quiet>'
    count = 0
    while count < 1000:
        try:
            key = next(ikeys)
        except StopIteration:
            break
        if isinstance(key, six.string_types):
            key_name = key
            version_id = None
        elif isinstance(key, tuple) and len(key) == 2:
            key_name, version_id = key
        elif (isinstance(key, Key) or isinstance(key, DeleteMarker)) and key.name:
            key_name = key.name
            version_id = key.version_id
        else:
            if isinstance(key, Prefix):
                key_name = key.name
                code = 'PrefixSkipped'
            else:
                key_name = repr(key)
                code = 'InvalidArgument'
            message = 'Invalid. No delete action taken for this object.'
            error = Error(key_name, code=code, message=message)
            result.errors.append(error)
            continue
        count += 1
        data += u'<Object><Key>%s</Key>' % xml.sax.saxutils.escape(key_name)
        if version_id:
            data += u'<VersionId>%s</VersionId>' % version_id
        data += u'</Object>'
    data += u'</Delete>'
    if count <= 0:
        return False
    data = data.encode('utf-8')
    fp = BytesIO(data)
    md5 = boto.utils.compute_md5(fp)
    hdrs['Content-MD5'] = md5[1]
    hdrs['Content-Type'] = 'text/xml'
    if mfa_token:
        hdrs[provider.mfa_header] = ' '.join(mfa_token)
    response = self.connection.make_request('POST', self.name, headers=hdrs, query_args=query_args, data=data)
    body = response.read()
    if response.status == 200:
        h = handler.XmlHandler(result, self)
        if not isinstance(body, bytes):
            body = body.encode('utf-8')
        xml.sax.parseString(body, h)
        return count >= 1000
    else:
        raise provider.storage_response_error(response.status, response.reason, body)