import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def common2fortran(common, tab=''):
    ret = ''
    for k in list(common.keys()):
        if k == '_BLNK_':
            ret = '%s%scommon %s' % (ret, tab, ','.join(common[k]))
        else:
            ret = '%s%scommon /%s/ %s' % (ret, tab, k, ','.join(common[k]))
    return ret