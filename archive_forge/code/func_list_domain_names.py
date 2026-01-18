import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def list_domain_names(self):
    """
        Lists all search domains owned by an account.
        """
    params = {}
    return self._make_request(action='ListDomainNames', verb='POST', path='/', params=params)