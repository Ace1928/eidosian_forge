from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def dicts_are_essentially_equal(a, b):
    """Make sure that a is a subset of b, where None entries of a are ignored."""
    for k, v in a.items():
        if v is None:
            continue
        if b.get(k) != v:
            return False
    return True