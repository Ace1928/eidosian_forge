from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
def sanitize_file_resource_for_windows(resource):
    """Returns the resource with invalid characters replaced.

  The invalid characters are only replaced if the resource URL is a FileUrl
  and the platform is Windows. This is required because Cloud URLs may have
  certain characters that are not allowed in file paths on Windows.

  Args:
    resource (Resource): The resource.

  Returns:
    The resource with invalid characters replaced from the path.
  """
    if not isinstance(resource.storage_url, storage_url.FileUrl) or not platforms.OperatingSystem.IsWindows() or (not properties.VALUES.storage.convert_incompatible_windows_path_characters.GetBool()):
        return resource
    sanitized_resource = copy.deepcopy(resource)
    sanitized_resource.storage_url.object_name = platforms.MakePathWindowsCompatible(sanitized_resource.storage_url.object_name)
    return sanitized_resource