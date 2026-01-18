import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def get_man_section(section):
    return os.path.join(get_manpath(), 'man%s' % section)