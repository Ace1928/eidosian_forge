from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.common.text.converters import to_native, to_bytes
def remove_old_versions(self):
    if not self.rolling_versions or self.versions_to_keep < 0:
        return
    if not self.check_mode:
        while len(self.configs) > max(self.versions_to_keep, 1):
            self.remove_config(self.configs.pop(0))