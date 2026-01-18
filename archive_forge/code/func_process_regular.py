from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def process_regular(in_path, tar, member):
    if not follow_links and os.path.exists(b_out_path):
        os.unlink(b_out_path)
    in_f = tar.extractfile(member)
    with open(b_out_path, 'wb') as out_f:
        shutil.copyfileobj(in_f, out_f)
    return in_path