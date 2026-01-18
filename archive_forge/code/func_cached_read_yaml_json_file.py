from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import yaml
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import files
import six
@function_result_cache.lru(maxsize=None)
def cached_read_yaml_json_file(file_path):
    """Converts JSON or YAML file to an in-memory dict.

  Args:
    file_path (str): Path for the file to parse passed in by the user.

  Returns:
    parsed (dict): Parsed value from the provided file_path.

  Raises:
    InvalidUrlError: The provided file_path either failed to load or be parsed
      as a dict.
  """
    expanded_file_path = os.path.realpath(os.path.expanduser(file_path))
    contents = files.ReadFileContents(expanded_file_path)
    return read_yaml_json_from_string(contents, source_path=file_path)