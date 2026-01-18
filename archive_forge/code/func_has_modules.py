import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def has_modules(self):
    return self.has_pure_modules() or self.has_ext_modules()