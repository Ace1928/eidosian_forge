import copy
import boto
import base64
import re
import six
from hashlib import md5
from boto.utils import compute_md5
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import write_to_fd
from boto.s3.prefix import Prefix
from boto.compat import six
def set_etag(self):
    """
        Set etag attribute by generating hex MD5 checksum on current
        contents of mock key.
        """
    m = md5()
    if not isinstance(self.data, bytes):
        m.update(self.data.encode('utf-8'))
    else:
        m.update(self.data)
    hex_md5 = m.hexdigest()
    self.etag = hex_md5