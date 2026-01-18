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
def get_standard_config_settings(self):
    settings = Values()
    for filename in self.get_standard_config_files():
        settings.update(self.get_config_file_settings(filename), self)
    return settings