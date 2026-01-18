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
@requires(['MarketplaceId'])
@api_action('Recommendations', 5, 2)
def get_last_updated_time_for_recommendations(self, request, response, **kw):
    """Checks whether there are active recommendations for each category
           for the given marketplace, and if there are, returns the time when
           recommendations were last updated for each category.
        """
    return self._post_request(request, kw, response)