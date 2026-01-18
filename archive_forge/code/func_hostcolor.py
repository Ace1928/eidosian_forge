from __future__ import (absolute_import, division, print_function)
import re
import sys
from ansible import constants as C
def hostcolor(host, stats, color=True):
    if ANSIBLE_COLOR and color:
        if stats['failures'] != 0 or stats['unreachable'] != 0:
            return u'%-37s' % stringc(host, C.COLOR_ERROR)
        elif stats['changed'] != 0:
            return u'%-37s' % stringc(host, C.COLOR_CHANGED)
        else:
            return u'%-37s' % stringc(host, C.COLOR_OK)
    return u'%-26s' % host