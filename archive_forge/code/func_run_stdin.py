import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def run_stdin(self, file_descriptor=sys.stdin, exit_on_error=True):
    """
        Reads commands to be run from a file descriptor, stdin by default.
        The run always starts from the root context.
        @param file_descriptor: The file descriptor to read commands from
        @type file_descriptor: file object
        @param exit_on_error: If True, stops the run if an error occurs
        @type exit_on_error: bool
        """
    self._current_node = self._root_node
    for cmdline in file_descriptor:
        try:
            self.run_cmdline(cmdline.strip())
        except Exception as msg:
            self.log.error(msg)
            if exit_on_error is True:
                raise ExecutionError('Aborting run on error.')
            self.log.exception('Keep running after an error.')