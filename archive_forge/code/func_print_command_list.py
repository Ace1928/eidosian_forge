import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def print_command_list(self, commands, header, max_length):
    """Print a subset of the list of all commands -- used by
        'print_commands()'.
        """
    print(header + ':')
    for cmd in commands:
        klass = self.cmdclass.get(cmd)
        if not klass:
            klass = self.get_command_class(cmd)
        try:
            description = klass.description
        except AttributeError:
            description = '(no description available)'
        print('  %-*s  %s' % (max_length, cmd, description))