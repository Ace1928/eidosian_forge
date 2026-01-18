from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
@requires(['MarketplaceId', 'Destination'])
@structured_objects('Destination', members=True)
@api_action('Subscriptions', 25, 0.5)
def send_test_notification_to_destination(self, request, response, **kw):
    """Sends a test notification to an existing destination.
        """
    return self._post_request(request, kw, response)