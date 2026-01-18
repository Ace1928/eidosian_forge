import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def get_command_packages(self):
    """Return a list of packages from which commands are loaded."""
    pkgs = self.command_packages
    if not isinstance(pkgs, list):
        if pkgs is None:
            pkgs = ''
        pkgs = [pkg.strip() for pkg in pkgs.split(',') if pkg != '']
        if 'distutils.command' not in pkgs:
            pkgs.insert(0, 'distutils.command')
        self.command_packages = pkgs
    return pkgs