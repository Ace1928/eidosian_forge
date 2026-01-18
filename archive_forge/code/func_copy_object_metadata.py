from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import datetime
import sys
from cloudsdk.google.protobuf import json_format
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util as json_metadata_util
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.util import crc32c
def copy_object_metadata(source_metadata, destination_metadata, request_config, should_deep_copy=False):
    """Copies specific metadata from source_metadata to destination_metadata.

  The API manually generates metadata for destination objects most of the time,
  but there are some fields that may not be populated.

  Args:
    source_metadata (gapic_clients.storage_v2.types.storage.Object): Metadata
      from source object.
    destination_metadata (gapic_clients.storage_v2.types.storage.Object):
      Metadata for destination object.
    request_config (request_config_factory.RequestConfig): Holds context info
      about the copy operation.
    should_deep_copy (bool): Copy all metadata, removing fields the backend must
      generate and preserving destination address.

  Returns:
    New destination metadata with data copied from source (messages.Object).
  """
    if should_deep_copy:
        destination_bucket = destination_metadata.bucket
        destination_name = destination_metadata.name
        destination_metadata = copy.deepcopy(source_metadata)
        destination_metadata.bucket = destination_bucket
        destination_metadata.name = destination_name
        destination_metadata.generation = None
        if request_config.resource_args.preserve_acl == False:
            destination_metadata.acl = []
    else:
        if request_config.resource_args.preserve_acl:
            if not source_metadata.acl:
                raise errors.Error('Attempting to preserve ACLs but found no source ACLs.')
            for source_acl in source_metadata.acl:
                destination_metadata.acl.append(copy.deepcopy(source_acl))
        destination_metadata.cache_control = source_metadata.cache_control
        destination_metadata.content_disposition = source_metadata.content_disposition
        destination_metadata.content_encoding = source_metadata.content_encoding
        destination_metadata.content_language = source_metadata.content_language
        destination_metadata.content_type = source_metadata.content_type
        if source_metadata.checksums:
            destination_metadata.checksums.crc32c = source_metadata.checksums.crc32c
            destination_metadata.checksums.md5_hash = source_metadata.checksums.md5_hash
        destination_metadata.custom_time = source_metadata.custom_time
        destination_metadata.metadata = copy.deepcopy(source_metadata.metadata)
    return destination_metadata