import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def set_canned_acl(self, acl_str, headers=None, generation=None, if_generation=None, if_metageneration=None):
    """Sets this objects's ACL using a predefined (canned) value.

        :type acl_str: string
        :param acl_str: A canned ACL string. See
            :data:`~.gs.acl.CannedACLStrings`.

        :type headers: dict
        :param headers: Additional headers to set during the request.

        :type generation: int
        :param generation: If specified, sets the ACL for a specific generation
            of a versioned object. If not specified, the current version is
            modified.

        :type if_generation: int
        :param if_generation: (optional) If set to a generation number, the acl
            will only be updated if its current generation number is this value.

        :type if_metageneration: int
        :param if_metageneration: (optional) If set to a metageneration number,
            the acl will only be updated if its current metageneration number is
            this value.
        """
    if self.bucket is not None:
        return self.bucket.set_canned_acl(acl_str, self.name, headers=headers, generation=generation, if_generation=if_generation, if_metageneration=if_metageneration)