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
def get_rewrite_token_from_tracker_file(tracker_file_path, rewrite_parameters_hash):
    """Attempts to read a rewrite tracker file.

  Args:
    tracker_file_path (str): The path to the tracker file.
    rewrite_parameters_hash (str): MD5 hex digest of rewrite call parameters
      constructed by hash_gcs_rewrite_parameters_for_tracker_file.

  Returns:
    String token for resuming rewrites if a matching tracker file exists.
  """
    if not os.path.exists(tracker_file_path):
        return None
    with files.FileReader(tracker_file_path) as tracker_file:
        existing_hash, rewrite_token = [line.rstrip('\n') for line in tracker_file.readlines()]
        if existing_hash == rewrite_parameters_hash:
            return rewrite_token
    return None