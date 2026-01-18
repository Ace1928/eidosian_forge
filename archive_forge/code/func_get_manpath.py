import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def get_manpath():
    manpath = 'share/man'
    if os.path.exists(os.path.join(sys.prefix, 'man')):
        manpath = 'man'
    return manpath