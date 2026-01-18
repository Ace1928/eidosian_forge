from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import string_types
def get_deprecation_messages():
    """Return a tuple of deprecations accumulated over this run"""
    return tuple(_global_deprecations)