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
def get_apitools_metadata_from_url(cloud_url):
    """Takes storage_url.CloudUrl and returns appropriate Apitools message."""
    messages = apis.GetMessagesModule('storage', 'v1')
    if cloud_url.is_bucket():
        return messages.Bucket(name=cloud_url.bucket_name)
    elif cloud_url.is_object():
        generation = int(cloud_url.generation) if cloud_url.generation else None
        return messages.Object(name=cloud_url.object_name, bucket=cloud_url.bucket_name, generation=generation)