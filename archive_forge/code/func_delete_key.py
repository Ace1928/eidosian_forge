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
def delete_key(self, key_name, headers=None, version_id=None, mfa_token=None, generation=None):
    """
        Deletes a key from the bucket.

        :type key_name: string
        :param key_name: The key name to delete

        :type headers: dict
        :param headers: A dictionary of header name/value pairs.

        :type version_id: string
        :param version_id: Unused in this subclass.

        :type mfa_token: tuple or list of strings
        :param mfa_token: Unused in this subclass.

        :type generation: int
        :param generation: The generation number of the key to delete. If not
            specified, the latest generation number will be deleted.

        :rtype: :class:`boto.gs.key.Key`
        :returns: A key object holding information on what was
            deleted.
        """
    query_args_l = []
    if generation:
        query_args_l.append('generation=%s' % generation)
    self._delete_key_internal(key_name, headers=headers, version_id=version_id, mfa_token=mfa_token, query_args_l=query_args_l)