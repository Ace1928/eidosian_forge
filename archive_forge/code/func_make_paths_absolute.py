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
def make_paths_absolute(pathdict, keys, base_path=None):
    """
    Interpret filesystem path settings relative to the `base_path` given.

    Paths are values in `pathdict` whose keys are in `keys`.  Get `keys` from
    `OptionParser.relative_path_settings`.
    """
    if base_path is None:
        base_path = os.getcwd()
    for key in keys:
        if key in pathdict:
            value = pathdict[key]
            if isinstance(value, list):
                value = [make_one_path_absolute(base_path, path) for path in value]
            elif value:
                value = make_one_path_absolute(base_path, value)
            pathdict[key] = value