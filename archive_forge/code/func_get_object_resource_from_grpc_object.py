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
def get_object_resource_from_grpc_object(grpc_object):
    """Returns the GCSObjectResource based off of the gRPC Object."""
    if grpc_object.generation is not None:
        generation = str(grpc_object.generation)
    else:
        generation = None
    url = storage_url.CloudUrl(scheme=storage_url.ProviderPrefix.GCS, bucket_name=grpc_object.bucket[GRPC_URL_BUCKET_OFFSET:], object_name=grpc_object.name, generation=generation)
    if grpc_object.customer_encryption and grpc_object.customer_encryption.key_sha256_bytes:
        decryption_key_hash_sha256 = hash_util.get_base64_string(grpc_object.customer_encryption.key_sha256_bytes)
        encryption_algorithm = grpc_object.customer_encryption.encryption_algorithm
    else:
        decryption_key_hash_sha256 = encryption_algorithm = None
    if grpc_object.checksums.crc32c is not None:
        crc32c_hash = crc32c.get_crc32c_hash_string_from_checksum(grpc_object.checksums.crc32c)
    else:
        crc32c_hash = None
    if grpc_object.checksums.md5_hash:
        md5_hash = hash_util.get_base64_string(grpc_object.checksums.md5_hash)
    else:
        md5_hash = None
    return gcs_resource_reference.GcsObjectResource(url, acl=_convert_repeated_message_to_dict(grpc_object.acl), cache_control=_get_value_or_none(grpc_object.cache_control), component_count=_get_value_or_none(grpc_object.component_count), content_disposition=_get_value_or_none(grpc_object.content_disposition), content_encoding=_get_value_or_none(grpc_object.content_encoding), content_language=_get_value_or_none(grpc_object.content_language), content_type=_get_value_or_none(grpc_object.content_type), crc32c_hash=crc32c_hash, creation_time=_convert_proto_to_datetime(grpc_object.create_time), custom_fields=_get_value_or_none(grpc_object.metadata), custom_time=_convert_proto_to_datetime(grpc_object.custom_time), decryption_key_hash_sha256=decryption_key_hash_sha256, encryption_algorithm=encryption_algorithm, etag=_get_value_or_none(grpc_object.etag), event_based_hold=grpc_object.event_based_hold if grpc_object.event_based_hold else None, kms_key=_get_value_or_none(grpc_object.kms_key), md5_hash=md5_hash, metadata=grpc_object, metageneration=grpc_object.metageneration, noncurrent_time=_convert_proto_to_datetime(grpc_object.delete_time), retention_expiration=_convert_proto_to_datetime(grpc_object.retention_expire_time), size=grpc_object.size, storage_class=_get_value_or_none(grpc_object.storage_class), storage_class_update_time=_convert_proto_to_datetime(grpc_object.update_storage_class_time), temporary_hold=grpc_object.temporary_hold if grpc_object.temporary_hold else None, update_time=_convert_proto_to_datetime(grpc_object.update_time))