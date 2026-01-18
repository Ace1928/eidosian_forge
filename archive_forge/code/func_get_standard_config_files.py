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
def get_standard_config_files(self):
    """Return list of config files, from environment or standard."""
    try:
        config_files = os.environ['DOCUTILSCONFIG'].split(os.pathsep)
    except KeyError:
        config_files = self.standard_config_files
    expand = os.path.expanduser
    if 'HOME' not in os.environ:
        try:
            import pwd
        except ImportError:
            expand = lambda x: x
    return [expand(f) for f in config_files if f.strip()]