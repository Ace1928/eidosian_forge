import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def confirm_private_virtual_interface(self, virtual_interface_id, virtual_gateway_id):
    """
        Accept ownership of a private virtual interface created by
        another customer.

        After the virtual interface owner calls this function, the
        virtual interface will be created and attached to the given
        virtual private gateway, and will be available for handling
        traffic.

        :type virtual_interface_id: string
        :param virtual_interface_id: ID of the virtual interface.
        Example: dxvif-123dfg56

        Default: None

        :type virtual_gateway_id: string
        :param virtual_gateway_id: ID of the virtual private gateway that will
            be attached to the virtual interface.
        A virtual private gateway can be managed via the Amazon Virtual Private
            Cloud (VPC) console or the `EC2 CreateVpnGateway`_ action.

        Default: None

        """
    params = {'virtualInterfaceId': virtual_interface_id, 'virtualGatewayId': virtual_gateway_id}
    return self.make_request(action='ConfirmPrivateVirtualInterface', body=json.dumps(params))