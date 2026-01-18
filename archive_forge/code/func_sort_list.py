from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def sort_list(unsorted_list):
    """
        Sort a given list.
        The list may contain dictionaries, so use the sort key to handle them.
        """
    if unsorted_list and isinstance(unsorted_list[0], dict):
        if not sort_key:
            raise Exception('A sort key was not specified when sorting list')
        else:
            return sorted(unsorted_list, key=lambda k: k[sort_key])
    try:
        return sorted(unsorted_list)
    except TypeError:
        return unsorted_list