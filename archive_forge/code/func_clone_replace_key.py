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
def clone_replace_key(self, key):
    return self.__class__(key.provider.get_provider_name(), bucket_name=key.bucket.name, object_name=key.name, suppress_consec_slashes=self.suppress_consec_slashes, version_id=getattr(key, 'version_id', None), generation=getattr(key, 'generation', None), is_latest=getattr(key, 'is_latest', None))