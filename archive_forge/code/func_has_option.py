import sys, string, re
import getopt
from distutils.errors import *
def has_option(self, long_option):
    """Return true if the option table for this parser has an
        option with long name 'long_option'."""
    return long_option in self.option_index