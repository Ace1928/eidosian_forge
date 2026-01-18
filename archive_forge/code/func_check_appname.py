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
def check_appname(appname):
    """Raise if the given sample app doesn't exist.

  Args:
    appname: str, Name of the sample app.

  Raises:
    ValueError: if the given sample app doesn't exist.
  """
    if appname not in APPS:
        raise ValueError("Unknown sample app '{}'".format(appname))