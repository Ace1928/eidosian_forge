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
def is_container_file_not_regular_file(container_stat):
    for bit in (32 - 1, 32 - 4, 32 - 5, 32 - 6, 32 - 7, 32 - 8, 32 - 11, 32 - 13):
        if container_stat['mode'] & 1 << bit != 0:
            return True
    return False