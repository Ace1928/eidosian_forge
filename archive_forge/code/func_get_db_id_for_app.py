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
def get_db_id_for_app(appname):
    """Get the database ID for the given sample app.

  Args:
    appname: str, Name of the sample app.

  Returns:
    str, The database ID, e.g. "finance-db".

  Raises:
    ValueError: if the given sample app doesn't exist.
  """
    check_appname(appname)
    return APPS[appname].db_id