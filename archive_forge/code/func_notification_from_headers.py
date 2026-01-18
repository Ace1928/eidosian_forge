from __future__ import absolute_import
import datetime
import uuid
from googleapiclient import errors
from googleapiclient import _helpers as util
import six
def notification_from_headers(channel, headers):
    """Parse a notification from the webhook request headers, validate
    the notification, and return a Notification object.

  Args:
    channel: Channel, The channel that the notification is associated with.
    headers: dict, A dictionary like object that contains the request headers
      from the webhook HTTP request.

  Returns:
    A Notification object.

  Raises:
    errors.InvalidNotificationError if the notification is invalid.
    ValueError if the X-GOOG-MESSAGE-NUMBER can't be converted to an int.
  """
    headers = _upper_header_keys(headers)
    channel_id = headers[X_GOOG_CHANNEL_ID]
    if channel.id != channel_id:
        raise errors.InvalidNotificationError('Channel id mismatch: %s != %s' % (channel.id, channel_id))
    else:
        message_number = int(headers[X_GOOG_MESSAGE_NUMBER])
        state = headers[X_GOOG_RESOURCE_STATE]
        resource_uri = headers[X_GOOG_RESOURCE_URI]
        resource_id = headers[X_GOOG_RESOURCE_ID]
        return Notification(message_number, state, resource_uri, resource_id)