from __future__ import (absolute_import, division, print_function)
from stringprep import (
from unicodedata import normalize
from ansible.module_utils.six import text_type
def is_unicode_str(string):
    return True if isinstance(string, text_type) else False