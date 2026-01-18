import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def get_all_topics(self, next_token=None):
    """
        :type next_token: string
        :param next_token: Token returned by the previous call to
                           this method.

        """
    params = {}
    if next_token:
        params['NextToken'] = next_token
    return self._make_request('ListTopics', params)