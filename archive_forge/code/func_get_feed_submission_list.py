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
@structured_lists('FeedSubmissionIdList.Id', 'FeedTypeList.Type', 'FeedProcessingStatusList.Status')
@api_action('Feeds', 10, 45)
def get_feed_submission_list(self, request, response, **kw):
    """Returns a list of all feed submissions submitted in the
           previous 90 days.
        """
    return self._post_request(request, kw, response)