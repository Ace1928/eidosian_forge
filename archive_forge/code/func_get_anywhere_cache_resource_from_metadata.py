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
def get_anywhere_cache_resource_from_metadata(metadata):
    url = storage_url.CloudUrl(scheme=storage_url.ProviderPrefix.GCS, bucket_name=metadata.bucket, object_name=metadata.anywhereCacheId)
    return gcs_resource_reference.GcsAnywhereCacheResource(admission_policy=metadata.admissionPolicy, anywhere_cache_id=metadata.anywhereCacheId, bucket=metadata.bucket, create_time=metadata.createTime, id_string=metadata.id, kind=metadata.kind, metadata=metadata, pending_update=metadata.pendingUpdate, state=metadata.state, storage_url=url, ttl=metadata.ttl, update_time=metadata.updateTime, zone=metadata.zone)