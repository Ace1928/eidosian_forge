import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def create_public_virtual_interface(self, connection_id, new_public_virtual_interface):
    """
        Creates a new public virtual interface. A virtual interface is
        the VLAN that transports AWS Direct Connect traffic. A public
        virtual interface supports sending traffic to public services
        of AWS such as Amazon Simple Storage Service (Amazon S3).

        :type connection_id: string
        :param connection_id: ID of the connection.
        Example: dxcon-fg5678gh

        Default: None

        :type new_public_virtual_interface: dict
        :param new_public_virtual_interface: Detailed information for the
            public virtual interface to be created.
        Default: None

        """
    params = {'connectionId': connection_id, 'newPublicVirtualInterface': new_public_virtual_interface}
    return self.make_request(action='CreatePublicVirtualInterface', body=json.dumps(params))