import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def create_interconnect(self, interconnect_name, bandwidth, location):
    """
        Creates a new interconnect between a AWS Direct Connect
        partner's network and a specific AWS Direct Connect location.

        An interconnect is a connection which is capable of hosting
        other connections. The AWS Direct Connect partner can use an
        interconnect to provide sub-1Gbps AWS Direct Connect service
        to tier 2 customers who do not have their own connections.
        Like a standard connection, an interconnect links the AWS
        Direct Connect partner's network to an AWS Direct Connect
        location over a standard 1 Gbps or 10 Gbps Ethernet fiber-
        optic cable. One end is connected to the partner's router, the
        other to an AWS Direct Connect router.

        For each end customer, the AWS Direct Connect partner
        provisions a connection on their interconnect by calling
        AllocateConnectionOnInterconnect. The end customer can then
        connect to AWS resources by creating a virtual interface on
        their connection, using the VLAN assigned to them by the AWS
        Direct Connect partner.

        :type interconnect_name: string
        :param interconnect_name: The name of the interconnect.
        Example: " 1G Interconnect to AWS "

        Default: None

        :type bandwidth: string
        :param bandwidth: The port bandwidth
        Example: 1Gbps

        Default: None

        Available values: 1Gbps,10Gbps

        :type location: string
        :param location: Where the interconnect is located
        Example: EqSV5

        Default: None

        """
    params = {'interconnectName': interconnect_name, 'bandwidth': bandwidth, 'location': location}
    return self.make_request(action='CreateInterconnect', body=json.dumps(params))