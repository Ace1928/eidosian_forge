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
@structured_lists('FeedSubmissionIdList.Id', 'FeedTypeList.Type')
@api_action('Feeds', 10, 45)
def cancel_feed_submissions(self, request, response, **kw):
    """Cancels one or more feed submissions and returns a
           count of the feed submissions that were canceled.
        """
    return self._post_request(request, kw, response)