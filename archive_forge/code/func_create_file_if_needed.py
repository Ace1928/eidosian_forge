from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import math
import os
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import scaled_integer
def create_file_if_needed(source_resource, destination_resource):
    """Creates new file if none exists or one that is too large exists at path.

  Args:
    source_resource (ObjectResource): Contains size metadata for target file.
    destination_resource(FileObjectResource|UnknownResource): Contains path to
      create file at.
  """
    file_path = destination_resource.storage_url.object_name
    if os.path.exists(file_path) and os.path.getsize(file_path) <= source_resource.size:
        return
    with files.BinaryFileWriter(file_path, create_path=True, mode=files.BinaryFileWriterMode.TRUNCATE, convert_invalid_windows_characters=properties.VALUES.storage.convert_incompatible_windows_path_characters.GetBool()):
        pass