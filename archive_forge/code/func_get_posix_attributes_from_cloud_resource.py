from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
def get_posix_attributes_from_cloud_resource(resource):
    """Parses metadata_dict and returns PosixAttributes.

  Note: This parses an object's *custom* metadata with user-set fields,
  not the full metadata with provider-set fields.

  Args:
    resource (ObjectResource): Contains URL to include in logged warnings and
      custom metadata to parse.

  Returns:
    PosixAttributes object populated from metadata_dict.
  """
    atime = _extract_time_from_custom_metadata(resource, ATIME_METADATA_KEY)
    mtime = _extract_time_from_custom_metadata(resource, MTIME_METADATA_KEY)
    uid = _extract_id_from_custom_metadata(resource, UID_METADATA_KEY)
    gid = _extract_id_from_custom_metadata(resource, GID_METADATA_KEY)
    mode = _extract_mode_from_custom_metadata(resource)
    return PosixAttributes(atime, mtime, uid, gid, mode)