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
def get_database_dialect(appname):
    """Get the database dialect for the given sample app.

  Args:
    appname: str, Name of the sample app.

  Returns:
    str, The database dialect.

  Raises:
    ValueError: if the given sample app doesn't exist.
  """
    check_appname(appname)
    return APPS[appname].database_dialect