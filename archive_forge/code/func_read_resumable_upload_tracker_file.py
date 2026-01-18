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
def read_resumable_upload_tracker_file(tracker_file_path):
    """Reads a resumable upload tracker file.

  Args:
    tracker_file_path (str): The path to the tracker file.

  Returns:
    A ResumableUploadTrackerData instance with data at tracker_file_path, or
    None if no file exists at tracker_file_path.
  """
    return _read_namedtuple_from_json_file(ResumableUploadTrackerData, tracker_file_path)