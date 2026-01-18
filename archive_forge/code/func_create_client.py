from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
import boto3
import botocore
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import headers_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.api_lib.storage import xml_metadata_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import s3transfer
def create_client(self, resource_location=None):
    """Creates the Boto3 client.

    Args:
      resource_location: (string) The name of the region associated with the
        client.

    Returns:
      A boto3 client.
    """
    disable_ssl_validation = properties.VALUES.auth.disable_ssl_validation.GetBool()
    if disable_ssl_validation:
        verify_ssl = False
    else:
        verify_ssl = None
    with BOTO3_CLIENT_LOCK:
        client = boto3.client(storage_url.ProviderPrefix.S3.value, aws_access_key_id=self.access_key_id, aws_secret_access_key=self.access_key_secret, region_name=resource_location, endpoint_url=self.endpoint_url, verify=verify_ssl)
        client.meta.events.register_first('before-sign.s3.*', self._add_additional_headers_to_request)
        return client