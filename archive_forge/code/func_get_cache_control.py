from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def get_cache_control(should_gzip_locally, resource_args):
    """Returns cache control metadata value.

  If should_gzip_locally is True, append 'no-transform' to cache control value
  with the user's given value.

  Args:
    should_gzip_locally (bool): True if file should be gzip locally.
    resource_args (request_config_factory._ObjectConfig): Holds settings for a
      cloud resource.

  Returns:
    (str|None) Cache control value.
  """
    if isinstance(resource_args, request_config_factory._ObjectConfig):
        user_cache_control = resource_args.cache_control
    else:
        user_cache_control = None
    if should_gzip_locally:
        return _NO_TRANSFORM if user_cache_control is None else '{}, {}'.format(user_cache_control, _NO_TRANSFORM)
    return user_cache_control