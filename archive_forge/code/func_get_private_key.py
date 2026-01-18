from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def get_private_key(self, private_key_path, **kwargs):
    return self._run_command([self._bin_path, '-l', '-f', private_key_path], **kwargs)