import boto
from boto.connection import AWSQueryConnection, AWSAuthConnection
from boto.exception import BotoServerError
import time
import urllib
import xml.sax
from boto.ecs.item import ItemSet
from boto import handler
def item_lookup(self, **params):
    """
        Returns items that satisfy the lookup query.

        For a full list of parameters, see:
        http://s3.amazonaws.com/awsdocs/Associates/2011-08-01/prod-adv-api-dg-2011-08-01.pdf
        """
    return self.get_response('ItemLookup', params)