from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import json
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer as apitools_transfer
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.api_lib.storage import headers_util
from googlecloudsdk.api_lib.storage.gcs_json import download
from googlecloudsdk.api_lib.storage.gcs_json import error_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.storage.gcs_json import patch_apitools_messages
from googlecloudsdk.api_lib.storage.gcs_json import upload
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
from six.moves import urllib
@error_util.catch_http_error_raise_gcs_api_error()
def bulk_restore_objects(self, bucket_url, object_globs, request_config, allow_overwrite=False, deleted_after_time=None, deleted_before_time=None):
    """See CloudApi class."""
    if request_config.resource_args:
        preserve_acl = request_config.resource_args.preserve_acl
    else:
        preserve_acl = None
    with self._apitools_request_headers_context({'x-goog-gcs-idempotency-token': uuid.uuid4().hex}):
        operation = self.client.objects.BulkRestore(self.messages.StorageObjectsBulkRestoreRequest(bucket=bucket_url.bucket_name, bulkRestoreObjectsRequest=self.messages.BulkRestoreObjectsRequest(allowOverwrite=allow_overwrite, copySourceAcl=preserve_acl, matchGlobs=object_globs, softDeletedAfterTime=deleted_after_time, softDeletedBeforeTime=deleted_before_time)))
    return operation