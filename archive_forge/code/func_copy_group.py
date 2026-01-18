import functools
import re
import sys
from Xlib.support import lock
def copy_group(group):
    return (copy_db(group[0]), copy_db(group[1])) + group[2:]