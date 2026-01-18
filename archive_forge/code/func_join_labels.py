from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.common.text.converters import to_text
def join_labels(labels, tail=''):
    """
    Combines the result of split_into_labels() back into a domain name.
    """
    return '.'.join(reversed(labels)) + tail