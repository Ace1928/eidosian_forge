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
@requires(['FeedSubmissionId'])
@api_action('Feeds', 15, 60)
def get_feed_submission_result(self, request, response, **kw):
    """Returns the feed processing report.
        """
    return self._post_request(request, kw, response)