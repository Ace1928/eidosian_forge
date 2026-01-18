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
def get_gcs_schema_name(appname):
    """Get the GCS file path for the schema for the given sample app.

  Doesn't include the bucket name. Use to download the sample app schema file
  from GCS.

  Args:
    appname: str, Name of the sample app.

  Returns:
    str, The sample app schema GCS file path.

  Raises:
    ValueError: if the given sample app doesn't exist.
  """
    check_appname(appname)
    app_attrs = APPS[appname]
    return '/'.join([app_attrs.gcs_prefix, _GCS_SCHEMA_PREFIX, app_attrs.schema_file])