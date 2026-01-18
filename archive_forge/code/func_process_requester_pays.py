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
def process_requester_pays(existing_billing, requester_pays):
    """Converts requester_pays boolean to Apitools object."""
    messages = apis.GetMessagesModule('storage', 'v1')
    if existing_billing:
        result_billing = existing_billing
    else:
        result_billing = messages.Bucket.BillingValue()
    result_billing.requesterPays = requester_pays
    return result_billing