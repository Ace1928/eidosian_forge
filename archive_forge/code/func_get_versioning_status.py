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
def get_versioning_status(self, headers=None):
    """Returns the current status of versioning configuration on the bucket.

        :rtype: bool
        """
    response = self.connection.make_request('GET', self.name, query_args='versioning', headers=headers)
    body = response.read()
    boto.log.debug(body)
    if response.status != 200:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)
    resp_json = boto.jsonresponse.Element()
    boto.jsonresponse.XmlHandler(resp_json, None).parse(body)
    resp_json = resp_json['VersioningConfiguration']
    return 'Status' in resp_json and resp_json['Status'] == 'Enabled'