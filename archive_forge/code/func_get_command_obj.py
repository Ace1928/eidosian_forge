import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def get_command_obj(self, command, create=1):
    """Return the command object for 'command'.  Normally this object
        is cached on a previous call to 'get_command_obj()'; if no command
        object for 'command' is in the cache, then we either create and
        return it (if 'create' is true) or return None.
        """
    cmd_obj = self.command_obj.get(command)
    if not cmd_obj and create:
        if DEBUG:
            self.announce("Distribution.get_command_obj(): creating '%s' command object" % command)
        klass = self.get_command_class(command)
        cmd_obj = self.command_obj[command] = klass(self)
        self.have_run[command] = 0
        options = self.command_options.get(command)
        if options:
            self._set_command_options(cmd_obj, options)
    return cmd_obj