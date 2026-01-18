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
def get_all_multipart_uploads(self, headers=None, **params):
    """
        A lower-level, version-aware method for listing active
        MultiPart uploads for a bucket.  This closely models the
        actual S3 API and requires you to manually handle the paging
        of results.  For a higher-level method that handles the
        details of paging for you, you can use the list method.

        :type max_uploads: int
        :param max_uploads: The maximum number of uploads to retrieve.
            Default value is 1000.

        :type key_marker: string
        :param key_marker: Together with upload_id_marker, this
            parameter specifies the multipart upload after which
            listing should begin.  If upload_id_marker is not
            specified, only the keys lexicographically greater than
            the specified key_marker will be included in the list.

            If upload_id_marker is specified, any multipart uploads
            for a key equal to the key_marker might also be included,
            provided those multipart uploads have upload IDs
            lexicographically greater than the specified
            upload_id_marker.

        :type upload_id_marker: string
        :param upload_id_marker: Together with key-marker, specifies
            the multipart upload after which listing should begin. If
            key_marker is not specified, the upload_id_marker
            parameter is ignored.  Otherwise, any multipart uploads
            for a key equal to the key_marker might be included in the
            list only if they have an upload ID lexicographically
            greater than the specified upload_id_marker.

        :type encoding_type: string
        :param encoding_type: Requests Amazon S3 to encode the response and
            specifies the encoding method to use.

            An object key can contain any Unicode character; however, XML 1.0
            parser cannot parse some characters, such as characters with an
            ASCII value from 0 to 10. For characters that are not supported in
            XML 1.0, you can add this parameter to request that Amazon S3
            encode the keys in the response.

            Valid options: ``url``

        :type delimiter: string
        :param delimiter: Character you use to group keys.
            All keys that contain the same string between the prefix, if
            specified, and the first occurrence of the delimiter after the
            prefix are grouped under a single result element, CommonPrefixes.
            If you don't specify the prefix parameter, then the substring
            starts at the beginning of the key. The keys that are grouped
            under CommonPrefixes result element are not returned elsewhere
            in the response.

        :type prefix: string
        :param prefix: Lists in-progress uploads only for those keys that
            begin with the specified prefix. You can use prefixes to separate
            a bucket into different grouping of keys. (You can think of using
            prefix to make groups in the same way you'd use a folder in a
            file system.)

        :rtype: ResultSet
        :return: The result from S3 listing the uploads requested

        """
    self.validate_kwarg_names(params, ['max_uploads', 'key_marker', 'upload_id_marker', 'encoding_type', 'delimiter', 'prefix'])
    return self._get_all([('Upload', MultiPartUpload), ('CommonPrefixes', Prefix)], 'uploads', headers, **params)