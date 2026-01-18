from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage.gcs_grpc import download
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import upload
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
def compose_objects(self, source_resources, destination_resource, request_config, original_source_resource=None, posix_to_set=None):
    """Concatenates a list of objects into a new object.

    Args:
      source_resources (list[ObjectResource|UnknownResource]): The objects to
        compose.
      destination_resource (resource_reference.UnknownResource): Metadata for
        the resulting composite object.
      request_config (RequestConfig): Object containing general API function
        arguments. Subclasses for specific cloud providers are available.
      original_source_resource (Resource|None): Useful for finding metadata to
        apply to final object. For instance, if doing a composite upload, this
        would represent the pre-split local file.
      posix_to_set (PosixAttributes|None): Set as custom metadata on target.

    Returns:
      resource_reference.ObjectResource with composite object's metadata.

    Raises:
      CloudApiError: API returned an error.
      NotImplementedError: This function was not implemented by a class using
        this interface.
    """
    if not source_resources:
        raise cloud_errors.GcsApiError('Compose requires at least one component object.')
    if len(source_resources) > self._MAX_OBJECTS_PER_COMPOSE_CALL:
        raise cloud_errors.GcsApiError(f'Compose was called with {len(source_resources)} objects. The limit is {self._MAX_OBJECTS_PER_COMPOSE_CALL}.')
    self._get_gapic_client()
    source_messages = []
    for source in source_resources:
        source_message = self._gapic_client.types.ComposeObjectRequest.SourceObject(name=source.storage_url.object_name)
        if source.storage_url.generation is not None:
            source_message.generation = int(source.storage_url.generation)
        source_messages.append(source_message)
    base_destination_metadata = metadata_util.get_grpc_metadata_from_url(destination_resource.storage_url, self._gapic_client.types)
    if getattr(source_resources[0], 'metadata', None) is not None:
        final_destination_metadata = metadata_util.copy_object_metadata(source_resources[0].metadata, base_destination_metadata, request_config)
    else:
        final_destination_metadata = base_destination_metadata
    metadata_util.update_object_metadata_from_request_config(final_destination_metadata, request_config, attributes_resource=original_source_resource, posix_to_set=posix_to_set)
    final_destination_metadata.bucket = destination_resource.storage_url.bucket_name
    final_destination_metadata.name = destination_resource.storage_url.object_name
    compose_request = self._gapic_client.types.ComposeObjectRequest(source_objects=source_messages, destination=final_destination_metadata, if_generation_match=request_config.precondition_generation_match, if_metageneration_match=request_config.precondition_metageneration_match)
    if request_config.resource_args:
        encryption_key = request_config.resource_args.encryption_key
        if encryption_key and encryption_key != user_request_args_factory.CLEAR and (encryption_key.type == encryption_util.KeyType.CMEK):
            compose_request.kms_key = encryption_key.key
    if request_config.predefined_acl_string is not None:
        compose_request.destination_predefined_acl = request_config.predefined_acl_string
    encryption_key = getattr(request_config.resource_args, 'encryption_key', None)
    with self._encryption_headers_context(encryption_key):
        return metadata_util.get_object_resource_from_grpc_object(self._gapic_client.storage.compose_object(compose_request))