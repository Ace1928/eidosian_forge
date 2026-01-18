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
def process_bucket_iam_configuration(existing_iam_metadata, public_access_prevention_boolean, uniform_bucket_level_access_boolean):
    """Converts user flags to Apitools IamConfigurationValue."""
    messages = apis.GetMessagesModule('storage', 'v1')
    if existing_iam_metadata:
        iam_metadata = existing_iam_metadata
    else:
        iam_metadata = messages.Bucket.IamConfigurationValue()
    if public_access_prevention_boolean is not None:
        if public_access_prevention_boolean:
            public_access_prevention_string = 'enforced'
        else:
            public_access_prevention_string = 'inherited'
        iam_metadata.publicAccessPrevention = public_access_prevention_string
    if uniform_bucket_level_access_boolean is not None:
        iam_metadata.uniformBucketLevelAccess = messages.Bucket.IamConfigurationValue.UniformBucketLevelAccessValue(enabled=uniform_bucket_level_access_boolean)
    return iam_metadata