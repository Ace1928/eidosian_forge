from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def run_proc(args, capture_logs_fn=None):
    """Wrapper for execution_utils.Subprocess that optionally captures logs.

  Args:
    args: [str], The arguments to execute.  The first argument is the command.
    capture_logs_fn: str, If set, save logs to the specified filename.

  Returns:
    subprocess.Popen or execution_utils.SubprocessTimeoutWrapper, The running
      subprocess.
  """
    if capture_logs_fn:
        logfile = files.FileWriter(capture_logs_fn, append=True, create_path=True)
        log.status.Print('Writing logs to {}'.format(capture_logs_fn))
        popen_args = dict(stdout=logfile, stderr=logfile)
    else:
        popen_args = {}
    return execution_utils.Subprocess(args, **popen_args)