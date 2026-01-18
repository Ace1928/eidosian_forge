from __future__ import (absolute_import, division, print_function)
import codecs
import re
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.quoting import unquote
def join_args(s):
    """
    Join the original cmd based on manipulations by split_args().
    This retains the original newlines and whitespaces.
    """
    result = ''
    for p in s:
        if len(result) == 0 or result.endswith('\n'):
            result += p
        else:
            result += ' ' + p
    return result