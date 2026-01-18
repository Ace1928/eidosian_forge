import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def run_commands(self):
    """Run each command that was seen on the setup script command line.
        Uses the list of commands found and cache of command objects
        created by 'get_command_obj()'.
        """
    for cmd in self.commands:
        self.run_command(cmd)