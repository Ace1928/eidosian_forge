from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import mimetypes
import os
import subprocess
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import buffered_upload_stream
from googlecloudsdk.command_lib.storage import component_stream
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import upload_stream
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def get_upload_strategy(api, object_length):
    """Determines if resumbale uplaod should be performed.

  Args:
    api (CloudApi): An api instance to check if it supports resumable upload.
    object_length (int): Length of the data to be uploaded.

  Returns:
    bool: True if resumable upload can be performed.
  """
    resumable_threshold = scaled_integer.ParseInteger(properties.VALUES.storage.resumable_threshold.Get())
    if object_length >= resumable_threshold and cloud_api.Capability.RESUMABLE_UPLOAD in api.capabilities:
        return cloud_api.UploadStrategy.RESUMABLE
    else:
        return cloud_api.UploadStrategy.SIMPLE