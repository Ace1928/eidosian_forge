from __future__ import absolute_import, division, print_function
import base64
import io
import os
import stat
import traceback
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.module_utils._scramble import generate_insecure_key, scramble
def are_fileobjs_equal(f1, f2):
    """Given two (buffered) file objects, compare their contents."""
    blocksize = 65536
    b1buf = b''
    b2buf = b''
    while True:
        if f1 and len(b1buf) < blocksize:
            f1b = f1.read(blocksize)
            if not f1b:
                f1 = None
            b1buf += f1b
        if f2 and len(b2buf) < blocksize:
            f2b = f2.read(blocksize)
            if not f2b:
                f2 = None
            b2buf += f2b
        if not b1buf or not b2buf:
            return not b1buf and (not b2buf)
        buflen = min(len(b1buf), len(b2buf))
        if b1buf[:buflen] != b2buf[:buflen]:
            return False
        b1buf = b1buf[buflen:]
        b2buf = b2buf[buflen:]