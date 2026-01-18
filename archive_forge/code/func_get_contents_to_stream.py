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
def get_contents_to_stream(self, fp, headers=NOT_IMPL, cb=NOT_IMPL, num_cb=NOT_IMPL, version_id=NOT_IMPL):
    key = self.get_key()
    return key.get_contents_to_file(fp)