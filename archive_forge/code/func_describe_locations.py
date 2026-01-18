import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def describe_locations(self):
    """
        Returns the list of AWS Direct Connect locations in the
        current AWS region. These are the locations that may be
        selected when calling CreateConnection or CreateInterconnect.
        """
    params = {}
    return self.make_request(action='DescribeLocations', body=json.dumps(params))