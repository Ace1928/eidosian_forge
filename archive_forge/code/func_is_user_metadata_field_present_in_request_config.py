from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import re
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.core import log
def is_user_metadata_field_present_in_request_config(request_config, attributes_resource=None, known_posix=None):
    """Checks the presence of user_metadata fields in request_config."""
    resource_args = request_config.resource_args
    if resource_args is None:
        return False
    if request_config.predefined_acl_string is not None:
        return True
    for value in _S3_TO_GENERIC_FIELD_NAMES_.values():
        if getattr(resource_args, value, None):
            return True
    return metadata_util.has_updated_custom_fields(resource_args, request_config.preserve_posix, request_config.preserve_symlinks, attributes_resource=attributes_resource, known_posix=known_posix)