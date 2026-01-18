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
def get_option_by_dest(self, dest):
    """
        Get an option by its dest.

        If you're supplying a dest which is shared by several options,
        it is undefined which option of those is returned.

        A KeyError is raised if there is no option with the supplied
        dest.
        """
    for group in self.option_groups + [self]:
        for option in group.option_list:
            if option.dest == dest:
                return option
    raise KeyError('No option with dest == %r.' % dest)