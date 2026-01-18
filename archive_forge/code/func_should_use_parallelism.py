from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import optimize_parameters_util
from googlecloudsdk.core import properties
def should_use_parallelism():
    """Checks execution settings to determine if parallelism should be used.

  This function is called in some tasks to determine how they are being
  executed, and should include as many of the relevant conditions as possible.

  Returns:
    True if parallel execution should be used, False otherwise.
  """
    process_count = properties.VALUES.storage.process_count.GetInt()
    thread_count = properties.VALUES.storage.thread_count.GetInt()
    if process_count is None or thread_count is None:
        return optimize_parameters_util.DEFAULT_TO_PARALLELISM
    return process_count > 1 or thread_count > 1