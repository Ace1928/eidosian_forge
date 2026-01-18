import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def describe_interconnects(self, interconnect_id=None):
    """
        Returns a list of interconnects owned by the AWS account.

        If an interconnect ID is provided, it will only return this
        particular interconnect.

        :type interconnect_id: string
        :param interconnect_id: The ID of the interconnect.
        Example: dxcon-abc123

        """
    params = {}
    if interconnect_id is not None:
        params['interconnectId'] = interconnect_id
    return self.make_request(action='DescribeInterconnects', body=json.dumps(params))