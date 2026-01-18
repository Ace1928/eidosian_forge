from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def replace_object_values_with_encryption_string(object_resource, encrypted_marker_string):
    """Updates fields to reflect that they are encrypted."""
    if object_resource.encryption_algorithm is None:
        return
    for key in ('md5_hash', 'crc32c_hash'):
        if getattr(object_resource, key) is None:
            setattr(object_resource, key, encrypted_marker_string)