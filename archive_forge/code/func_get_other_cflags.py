import sys, os, subprocess
from .error import PkgConfigError
def get_other_cflags(string):
    return [x for x in string.split() if not x.startswith('-I') and (not x.startswith('-D'))]