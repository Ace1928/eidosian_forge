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
def setmesstext(block):
    global filepositiontext
    try:
        filepositiontext = 'In: %s:%s\n' % (block['from'], block['name'])
    except Exception:
        pass