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
def set_def_canned_acl(self, acl_str, headers=None):
    """Sets a bucket's default ACL using a predefined (canned) value.

        :type acl_str: string
        :param acl_str: A canned ACL string. See
            :data:`~.gs.acl.CannedACLStrings`.

        :type headers: dict
        :param headers: Additional headers to set during the request.
        """
    if acl_str not in CannedACLStrings:
        raise ValueError('Provided canned ACL string (%s) is not valid.' % acl_str)
    query_args = DEF_OBJ_ACL
    return self._set_acl_helper(acl_str, '', headers, query_args, generation=None, if_generation=None, if_metageneration=None, canned=True)