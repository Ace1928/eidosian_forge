from __future__ import absolute_import
import datetime
import uuid
from googleapiclient import errors
from googleapiclient import _helpers as util
import six
@util.positional(2)
def new_webhook_channel(url, token=None, expiration=None, params=None):
    """Create a new webhook Channel.

    Args:
      url: str, URL to post notifications to.
      token: str, An arbitrary string associated with the channel that
        is delivered to the target address with each notification delivered
        over this channel.
      expiration: datetime.datetime, A time in the future when the channel
        should expire. Can also be None if the subscription should use the
        default expiration. Note that different services may have different
        limits on how long a subscription lasts. Check the response from the
        watch() method to see the value the service has set for an expiration
        time.
      params: dict, Extra parameters to pass on channel creation. Currently
        not used for webhook channels.
    """
    expiration_ms = 0
    if expiration:
        delta = expiration - EPOCH
        expiration_ms = delta.microseconds / 1000 + (delta.seconds + delta.days * 24 * 3600) * 1000
        if expiration_ms < 0:
            expiration_ms = 0
    return Channel('web_hook', str(uuid.uuid4()), token, url, expiration=expiration_ms, params=params)