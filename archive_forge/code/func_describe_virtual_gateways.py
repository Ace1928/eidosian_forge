import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def describe_virtual_gateways(self):
    """
        Returns a list of virtual private gateways owned by the AWS
        account.

        You can create one or more AWS Direct Connect private virtual
        interfaces linking to a virtual private gateway. A virtual
        private gateway can be managed via Amazon Virtual Private
        Cloud (VPC) console or the `EC2 CreateVpnGateway`_ action.
        """
    params = {}
    return self.make_request(action='DescribeVirtualGateways', body=json.dumps(params))