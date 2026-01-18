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
def get_notification_configuration_iterator(urls, accept_notification_configuration_urls=True):
    """Yields bucket/notification tuples from command-line args.

  Given a list of strings that are bucket URLs ("gs://foo") or notification
  configuration URLs ("b/bucket/notificationConfigs/5"), yield tuples of
  bucket names and their associated notifications.

  Args:
    urls (list[str]): Bucket and notification configuration URLs to pull
      notification configurations from.
    accept_notification_configuration_urls (bool): Whether to raise an an error
      if a notification configuration URL is in `urls`.

  Yields:
    NotificationIteratorResult

  Raises:
    InvalidUrlError: Received notification configuration URL, but
      accept_notification_configuration_urls was False. Or received non-GCS
      bucket URL.
  """
    client = api_factory.get_api(storage_url.ProviderPrefix.GCS)
    for url in urls:
        bucket_url, notification_id = get_bucket_url_and_notification_id_from_url(url)
        if notification_id:
            if not accept_notification_configuration_urls:
                raise errors.InvalidUrlError('Received disallowed notification configuration URL: ' + url)
            notification_configuration = client.get_notification_configuration(bucket_url, notification_id)
            yield NotificationIteratorResult(bucket_url, notification_configuration)
        else:
            cloud_url = storage_url.storage_url_from_string(url)
            raise_error_if_not_gcs_bucket_matching_url(cloud_url)
            if cloud_url.is_provider():
                bucket_url = storage_url.CloudUrl(storage_url.ProviderPrefix.GCS, '*')
            else:
                bucket_url = cloud_url
            for bucket_resource in wildcard_iterator.get_wildcard_iterator(bucket_url.url_string, fields_scope=cloud_api.FieldsScope.SHORT):
                for notification_configuration in client.list_notification_configurations(bucket_resource.storage_url):
                    yield NotificationIteratorResult(bucket_resource.storage_url, notification_configuration)