import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def run_cmdline(self, cmdline):
    """
        Runs the specified command. Global commands are checked first,
        then local commands from the current node.

        Command syntax is:
        [PATH] COMMAND [POSITIONAL_PARAMETER]+ [PARAMETER=VALUE]+

        @param cmdline: The command line to run
        @type cmdline: str
        """
    if cmdline:
        self.log.verbose("Running command line '%s'." % cmdline)
        path, command, pparams, kparams = self._parse_cmdline(cmdline)[1:]
        self._execute_command(path, command, pparams, kparams)