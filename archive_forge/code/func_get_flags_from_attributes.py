from __future__ import (absolute_import, division, print_function)
import os
import stat
import re
def get_flags_from_attributes(attributes):
    flags = [key for key, attr in FILE_ATTRIBUTES.items() if attr in attributes]
    return ''.join(flags)