import os
import os.path
import sys
import warnings
import configparser as CP
import codecs
import optparse
from optparse import SUPPRESS_HELP
import docutils
import docutils.utils
import docutils.nodes
from docutils.utils.error_reporting import (locale_encoding, SafeString,
def store_multiple(option, opt, value, parser, *args, **kwargs):
    """
    Store multiple values in `parser.values`.  (Option callback.)

    Store `None` for each attribute named in `args`, and store the value for
    each key (attribute name) in `kwargs`.
    """
    for attribute in args:
        setattr(parser.values, attribute, None)
    for key, value in list(kwargs.items()):
        setattr(parser.values, key, value)