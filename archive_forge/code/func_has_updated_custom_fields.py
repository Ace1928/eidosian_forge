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
def has_updated_custom_fields(resource_args, preserve_posix, preserve_symlinks, attributes_resource=None, known_posix=None):
    """Checks for the storage provider's custom metadata field."""
    file_resource = isinstance(attributes_resource, resource_reference.FileObjectResource)
    should_parse_file_posix = preserve_posix and file_resource
    should_parse_symlinks = preserve_symlinks and file_resource
    return bool(should_parse_file_posix or known_posix or should_parse_symlinks or resource_args.custom_fields_to_set or resource_args.custom_fields_to_remove or resource_args.custom_fields_to_update)