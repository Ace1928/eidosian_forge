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
@requires(['FeedType'])
@boolean_arguments('PurgeAndReplace')
@http_body('FeedContent')
@structured_lists('MarketplaceIdList.Id')
@api_action('Feeds', 15, 120)
def submit_feed(self, request, response, headers=None, body='', **kw):
    """Uploads a feed for processing by Amazon MWS.
        """
    headers = headers or {}
    return self._post_request(request, kw, response, body=body, headers=headers)