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
def get_container_file_mode(container_stat):
    mode = container_stat['mode'] & 4095
    if container_stat['mode'] & 1 << 32 - 9 != 0:
        mode |= stat.S_ISUID
    if container_stat['mode'] & 1 << 32 - 10 != 0:
        mode |= stat.S_ISGID
    if container_stat['mode'] & 1 << 32 - 12 != 0:
        mode |= stat.S_ISVTX
    return mode