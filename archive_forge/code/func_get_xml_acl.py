import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def get_xml_acl(self, headers=None, generation=None):
    """Returns the ACL string of this object.

        :param dict headers: Additional headers to set during the request.

        :param int generation: If specified, gets the ACL for a specific
            generation of a versioned object. If not specified, the current
            version is returned.

        :rtype: str
        """
    if self.bucket is not None:
        return self.bucket.get_xml_acl(self.name, headers=headers, generation=generation)