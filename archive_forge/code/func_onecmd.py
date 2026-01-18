from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd
import shlex
from typing import List, Optional
from absl import flags
from pyglib import appcommands
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
def onecmd(self, line: str) -> bool:
    """Process a single command.

    Runs a single command, and stores the return code in
    self._last_return_code. Always returns False unless the command
    was EOF.

    Args:
      line: (str) Command line to process.

    Returns:
      A bool signaling whether or not the command loop should terminate.
    """
    try:
        self._last_return_code = cmd.Cmd.onecmd(self, line)
    except CommandLoop.TerminateSignal:
        return True
    except BaseException as e:
        name = line.split(' ')[0]
        bq_utils.ProcessError(e, name=name)
        self._last_return_code = 1
    return False