from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import sys
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import iso_duration
def process_log_config(target_bucket, log_bucket, log_object_prefix):
    """Converts log setting to Apitools object.

  Args:
    target_bucket (str): Bucket to track with logs.
    log_bucket (str|None): Bucket to store logs in.
    log_object_prefix (str|None): Prefix for objects to create logs for.

  Returns:
    messages.Bucket.LoggingValue: Apitools log settings object.
  """
    if log_bucket in ('', None, user_request_args_factory.CLEAR):
        return None
    messages = apis.GetMessagesModule('storage', 'v1')
    logging_value = messages.Bucket.LoggingValue()
    logging_value.logBucket = storage_url.remove_scheme(log_bucket)
    if log_object_prefix == user_request_args_factory.CLEAR:
        logging_value.logObjectPrefix = None
    else:
        logging_value.logObjectPrefix = storage_url.remove_scheme(log_object_prefix or target_bucket)
    return logging_value