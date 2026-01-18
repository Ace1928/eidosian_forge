from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def is_masklen(val):
    try:
        return 0 <= int(val) <= 32
    except ValueError:
        return False