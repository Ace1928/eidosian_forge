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
def get_tracker_file_path(destination_url, tracker_file_type, source_url=None, component_number=None):
    """Retrieves path string to tracker file.

  Args:
    destination_url (storage_url.StorageUrl): Describes the destination file.
    tracker_file_type (TrackerFileType): Type of tracker file to retrieve.
    source_url (storage_url.StorageUrl): Describes the source file.
    component_number (int): The number of the component is being tracked for a
      sliced download or composite upload.

  Returns:
    String file path to tracker file.
  """
    if tracker_file_type is TrackerFileType.UPLOAD:
        if component_number is not None:
            object_name, _, _ = destination_url.object_name.rpartition('_')
        else:
            object_name = destination_url.object_name
        raw_result_tracker_file_name = 'resumable_upload__{}__{}__{}.url'.format(destination_url.bucket_name, object_name, destination_url.scheme.value)
    elif tracker_file_type is TrackerFileType.DOWNLOAD:
        raw_result_tracker_file_name = 'resumable_download__{}__{}.etag'.format(os.path.realpath(destination_url.object_name), destination_url.scheme.value)
    elif tracker_file_type is TrackerFileType.DOWNLOAD_COMPONENT:
        raw_result_tracker_file_name = 'resumable_download__{}__{}__{}.etag'.format(os.path.realpath(destination_url.object_name), destination_url.scheme.value, component_number)
    elif tracker_file_type is TrackerFileType.PARALLEL_UPLOAD:
        raw_result_tracker_file_name = 'parallel_upload__{}__{}__{}__{}.url'.format(destination_url.bucket_name, destination_url.object_name, source_url, destination_url.scheme.value)
    elif tracker_file_type is TrackerFileType.SLICED_DOWNLOAD:
        raw_result_tracker_file_name = 'sliced_download__{}__{}.etag'.format(os.path.realpath(destination_url.object_name), destination_url.scheme.value)
    elif tracker_file_type is TrackerFileType.REWRITE:
        raw_result_tracker_file_name = 'rewrite__{}__{}__{}__{}__{}.token'.format(source_url.bucket_name, source_url.object_name, destination_url.bucket_name, destination_url.object_name, destination_url.scheme.value)
    result_tracker_file_name = get_delimiterless_file_path(raw_result_tracker_file_name)
    resumable_tracker_directory = _create_tracker_directory_if_needed()
    return _get_hashed_tracker_file_path(result_tracker_file_name, tracker_file_type, resumable_tracker_directory, component_number)