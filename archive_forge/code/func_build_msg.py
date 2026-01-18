from __future__ import absolute_import, division, print_function
import errno
import json
import os
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.image_archive import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import (
from ansible_collections.community.docker.plugins.module_utils._api.constants import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils._api.utils.build import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def build_msg(reason):
    return 'Archived image %s to %s, %s' % (current_image_name, archive_path, reason)