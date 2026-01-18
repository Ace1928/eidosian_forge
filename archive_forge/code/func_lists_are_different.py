from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def lists_are_different(list1, list2):
    diff = False
    if sorted(list1) != sorted(list2):
        diff = True
    return diff