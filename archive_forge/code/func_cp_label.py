from __future__ import (absolute_import, division, print_function)
import re
def cp_label(value):
    p = re.compile('[^-\\w]+')
    return p.sub('_', value)