from __future__ import (absolute_import, division, print_function)
import re
def compare_platform_strings(string1, string2):
    return _Platform.parse_platform_string(string1) == _Platform.parse_platform_string(string2)