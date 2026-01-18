import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
def load_bytecode(self, bucket: Bucket) -> None:
    try:
        code = self.client.get(self.prefix + bucket.key)
    except Exception:
        if not self.ignore_memcache_errors:
            raise
    else:
        bucket.bytecode_from_string(code)