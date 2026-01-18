from __future__ import absolute_import
import re
import sys
def replace_specials(m):
    return replacements[m.group(1)]