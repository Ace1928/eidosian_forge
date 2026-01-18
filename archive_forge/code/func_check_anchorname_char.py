from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
def check_anchorname_char(ch):
    if ch in u',[]{}':
        return False
    return check_namespace_char(ch)