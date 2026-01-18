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
def get_bucket_metadata_dict_from_request_config(request_config):
    """Returns S3 bucket metadata dict fields based on RequestConfig."""
    metadata = {}
    resource_args = request_config.resource_args
    if resource_args:
        if resource_args.cors_file_path is not None:
            metadata.update(xml_metadata_field_converters.process_cors(resource_args.cors_file_path))
        if resource_args.labels_file_path is not None:
            metadata.update(xml_metadata_field_converters.process_labels(resource_args.labels_file_path))
        if resource_args.lifecycle_file_path is not None:
            metadata.update(xml_metadata_field_converters.process_lifecycle(resource_args.lifecycle_file_path))
        if resource_args.location is not None:
            metadata['LocationConstraint'] = resource_args.location
        if resource_args.requester_pays is not None:
            metadata.update(xml_metadata_field_converters.process_requester_pays(resource_args.requester_pays))
        if resource_args.versioning is not None:
            metadata.update(xml_metadata_field_converters.process_versioning(resource_args.versioning))
        if resource_args.web_error_page is not None or resource_args.web_main_page_suffix is not None:
            metadata.update(xml_metadata_field_converters.process_website(resource_args.web_error_page, resource_args.web_main_page_suffix))
    return metadata