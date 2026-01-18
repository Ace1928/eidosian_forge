import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def confirm_public_virtual_interface(self, virtual_interface_id):
    """
        Accept ownership of a public virtual interface created by
        another customer.

        After the virtual interface owner calls this function, the
        specified virtual interface will be created and made available
        for handling traffic.

        :type virtual_interface_id: string
        :param virtual_interface_id: ID of the virtual interface.
        Example: dxvif-123dfg56

        Default: None

        """
    params = {'virtualInterfaceId': virtual_interface_id}
    return self.make_request(action='ConfirmPublicVirtualInterface', body=json.dumps(params))