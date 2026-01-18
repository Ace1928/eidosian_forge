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
def get_local_bin_path(appname):
    """Get the local path to binaries for the given sample app.

  This typically includes server and workload binaries and any required
  dependencies. Note that the path may not exist.

  Args:
    appname: str, Name of the sample app.

  Returns:
    str, The local path of the sample app binaries.

  Raises:
    ValueError: if the given sample app doesn't exist.
  """
    check_appname(appname)
    return os.path.join(SAMPLES_BIN_PATH, APPS[appname].bin_path)