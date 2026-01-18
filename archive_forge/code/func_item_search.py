import boto
from boto.connection import AWSQueryConnection, AWSAuthConnection
from boto.exception import BotoServerError
import time
import urllib
import xml.sax
from boto.ecs.item import ItemSet
from boto import handler
def item_search(self, search_index, **params):
    """
        Returns items that satisfy the search criteria, including one or more search
        indices.

        For a full list of search terms,
        :see: http://docs.amazonwebservices.com/AWSECommerceService/2010-09-01/DG/index.html?ItemSearch.html
        """
    params['SearchIndex'] = search_index
    return self.get_response('ItemSearch', params)