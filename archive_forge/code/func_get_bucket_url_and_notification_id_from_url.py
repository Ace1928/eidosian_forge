from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
def get_bucket_url_and_notification_id_from_url(url_string):
    """Extracts bucket StorageUrl and notification_id string from URL."""
    match = _CANONICAL_NOTIFICATION_CONFIGURATION_REGEX.match(url_string) or _JSON_NOTIFICATION_CONFIGURATION_REGEX.match(url_string)
    if match:
        return (storage_url.CloudUrl(storage_url.ProviderPrefix.GCS, match.group('bucket_name')), match.group('notification_id'))
    return (None, None)