import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def cache_hash(self, *factors):
    chash = 0
    for f in factors:
        for char in str(f):
            chash = ord(char) + (chash << 6) + (chash << 16) - chash
            chash &= 4294967295
    return chash