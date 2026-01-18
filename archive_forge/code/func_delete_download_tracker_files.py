from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def delete_download_tracker_files(destination_url):
    """Deletes all tracker files for an object download.

  Deletes files for different strategies in case download was interrupted and
  resumed with a different strategy. Prevents orphaned tracker files.

  Args:
    destination_url (storage_url.StorageUrl): Describes the destination file.
  """
    sliced_download_tracker_files = _get_sliced_download_tracker_file_paths(destination_url)
    for tracker_file in sliced_download_tracker_files:
        delete_tracker_file(tracker_file)
    delete_tracker_file(get_tracker_file_path(destination_url, TrackerFileType.DOWNLOAD))