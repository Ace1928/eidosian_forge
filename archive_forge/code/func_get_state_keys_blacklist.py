from .error import *
from .nodes import *
import collections.abc, datetime, base64, binascii, re, sys, types
def get_state_keys_blacklist(self):
    return ['^extend$', '^__.*__$']