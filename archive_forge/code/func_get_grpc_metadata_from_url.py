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
def get_grpc_metadata_from_url(cloud_url, grpc_types):
    """Takes storage_url.CloudUrl and returns appropriate Types message."""
    if cloud_url.is_bucket():
        return grpc_types.Bucket(name=cloud_url.bucket_name)
    elif cloud_url.is_object():
        generation = int(cloud_url.generation) if cloud_url.generation else None
        return grpc_types.Object(name=cloud_url.object_name, bucket=cloud_url.bucket_name, generation=generation)