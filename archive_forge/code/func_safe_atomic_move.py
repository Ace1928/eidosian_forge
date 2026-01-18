from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@restore_on_failure
def safe_atomic_move(module, path, destination):
    module.atomic_move(path, destination)