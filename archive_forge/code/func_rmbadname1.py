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
def rmbadname1(name):
    if name in badnames:
        errmess('rmbadname1: Replacing "%s" with "%s".\n' % (name, badnames[name]))
        return badnames[name]
    return name