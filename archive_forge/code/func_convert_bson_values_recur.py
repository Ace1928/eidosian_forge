from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def convert_bson_values_recur(mydict):
    """
    Converts values that Ansible doesn't like
    # https://github.com/ansible-collections/community.mongodb/issues/462
    """
    if isinstance(mydict, dict):
        for key, value in mydict.items():
            if isinstance(value, dict):
                mydict[key] = convert_bson_values_recur(value)
            elif isinstance(value, TYPES_NEED_TO_CONVERT):
                mydict[key] = convert_to_supported(value)
            else:
                mydict[key] = value
    return mydict