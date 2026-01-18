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
def get_object_resource_from_metadata(metadata):
    """Helper method to generate a ObjectResource instance from GCS metadata.

  Args:
    metadata (messages.Object): Extract resource properties from this.

  Returns:
    ObjectResource with properties populated by metadata.
  """
    if metadata.generation is not None:
        generation = str(metadata.generation)
    else:
        generation = None
    url = storage_url.CloudUrl(scheme=storage_url.ProviderPrefix.GCS, bucket_name=metadata.bucket, object_name=metadata.name, generation=generation)
    if metadata.customerEncryption:
        decryption_key_hash_sha256 = metadata.customerEncryption.keySha256
        encryption_algorithm = metadata.customerEncryption.encryptionAlgorithm
    else:
        decryption_key_hash_sha256 = encryption_algorithm = None
    return gcs_resource_reference.GcsObjectResource(url, acl=_message_to_dict(metadata.acl), cache_control=metadata.cacheControl, component_count=metadata.componentCount, content_disposition=metadata.contentDisposition, content_encoding=metadata.contentEncoding, content_language=metadata.contentLanguage, content_type=metadata.contentType, crc32c_hash=metadata.crc32c, creation_time=metadata.timeCreated, custom_fields=_message_to_dict(metadata.metadata), custom_time=metadata.customTime, decryption_key_hash_sha256=decryption_key_hash_sha256, encryption_algorithm=encryption_algorithm, etag=metadata.etag, event_based_hold=metadata.eventBasedHold if metadata.eventBasedHold else None, hard_delete_time=metadata.hardDeleteTime, kms_key=metadata.kmsKeyName, md5_hash=metadata.md5Hash, metadata=metadata, metageneration=metadata.metageneration, noncurrent_time=metadata.timeDeleted, retention_expiration=metadata.retentionExpirationTime, retention_settings=_message_to_dict(metadata.retention), size=metadata.size, soft_delete_time=metadata.softDeleteTime, storage_class=metadata.storageClass, storage_class_update_time=metadata.timeStorageClassUpdated, temporary_hold=metadata.temporaryHold if metadata.temporaryHold else None, update_time=metadata.updated)