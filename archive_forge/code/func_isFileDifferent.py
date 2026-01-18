from __future__ import print_function
import sys
import os
import platform
import io
import getopt
import re
import string
import errno
import copy
import glob
from jsbeautifier.__version__ import __version__
from jsbeautifier.javascript.options import BeautifierOptions
from jsbeautifier.javascript.beautifier import Beautifier
def isFileDifferent(filepath, expected):
    try:
        return ''.join(io.open(filepath, 'rt', newline='').readlines()) != expected
    except BaseException:
        return True