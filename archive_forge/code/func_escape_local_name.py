from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
def escape_local_name(name):
    return name.replace(_ESCAPE_CHAR, _ESCAPE_CHAR + _ESCAPE_CHAR).replace('/', _ESCAPE_CHAR + 'S')