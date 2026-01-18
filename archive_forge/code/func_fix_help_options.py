import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def fix_help_options(options):
    """Convert a 4-tuple 'help_options' list as found in various command
    classes to the 3-tuple form required by FancyGetopt.
    """
    new_options = []
    for help_tuple in options:
        new_options.append(help_tuple[0:3])
    return new_options