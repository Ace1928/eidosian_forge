from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import xml.sax
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import GSResponseError
from boto.exception import InvalidAclError
from boto.gs.acl import ACL, CannedACLStrings
from boto.gs.acl import SupportedPermissions as GSPermissions
from boto.gs.bucketlistresultset import VersionedBucketListResultSet
from boto.gs.cors import Cors
from boto.gs.encryptionconfig import EncryptionConfig
from boto.gs.lifecycle import LifecycleConfig
from boto.gs.key import Key as GSKey
from boto.s3.acl import Policy
from boto.s3.bucket import Bucket as S3Bucket
from boto.utils import get_utf8able_str
from boto.compat import quote
from boto.compat import six
def get_storage_class(self, headers=None):
    """
        Returns the StorageClass for the bucket.

        :rtype: str
        :return: The StorageClass for the bucket.
        """
    response = self.connection.make_request('GET', self.name, query_args=STORAGE_CLASS_ARG, headers=headers)
    body = response.read()
    if response.status == 200:
        rs = ResultSet(self)
        h = handler.XmlHandler(rs, self)
        xml.sax.parseString(body, h)
        return rs.StorageClass
    else:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)