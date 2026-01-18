import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def parse_command_line(self):
    """Parse the setup script's command line, taken from the
        'script_args' instance attribute (which defaults to 'sys.argv[1:]'
        -- see 'setup()' in core.py).  This list is first processed for
        "global options" -- options that set attributes of the Distribution
        instance.  Then, it is alternately scanned for Distutils commands
        and options for that command.  Each new command terminates the
        options for the previous command.  The allowed options for a
        command are determined by the 'user_options' attribute of the
        command class -- thus, we have to be able to load command classes
        in order to parse the command line.  Any error in that 'options'
        attribute raises DistutilsGetoptError; any error on the
        command-line raises DistutilsArgError.  If no Distutils commands
        were found on the command line, raises DistutilsArgError.  Return
        true if command-line was successfully parsed and we should carry
        on with executing commands; false if no errors but we shouldn't
        execute commands (currently, this only happens if user asks for
        help).
        """
    toplevel_options = self._get_toplevel_options()
    self.commands = []
    parser = FancyGetopt(toplevel_options + self.display_options)
    parser.set_negative_aliases(self.negative_opt)
    parser.set_aliases({'licence': 'license'})
    args = parser.getopt(args=self.script_args, object=self)
    option_order = parser.get_option_order()
    log.set_verbosity(self.verbose)
    if self.handle_display_options(option_order):
        return
    while args:
        args = self._parse_command_opts(parser, args)
        if args is None:
            return
    if self.help:
        self._show_help(parser, display_options=len(self.commands) == 0, commands=self.commands)
        return
    if not self.commands:
        raise DistutilsArgError('no commands supplied')
    return True