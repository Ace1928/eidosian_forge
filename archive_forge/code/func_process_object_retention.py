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
def process_object_retention(existing_retention_settings, retain_until, retention_mode):
    """Converts individual object retention settings to Apitools object."""
    if retain_until == user_request_args_factory.CLEAR or retention_mode == user_request_args_factory.CLEAR or (not any([existing_retention_settings, retain_until, retention_mode])):
        return None
    if existing_retention_settings is None:
        messages = apis.GetMessagesModule('storage', 'v1')
        retention_settings = messages.Object.RetentionValue()
    else:
        retention_settings = existing_retention_settings
    if retain_until:
        retention_settings.retainUntilTime = retain_until
    if retention_mode:
        retention_settings.mode = retention_mode.value
    return retention_settings