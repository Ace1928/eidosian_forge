import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def allocate_public_virtual_interface(self, connection_id, owner_account, new_public_virtual_interface_allocation):
    """
        Provisions a public virtual interface to be owned by a
        different customer.

        The owner of a connection calls this function to provision a
        public virtual interface which will be owned by another AWS
        customer.

        Virtual interfaces created using this function must be
        confirmed by the virtual interface owner by calling
        ConfirmPublicVirtualInterface. Until this step has been
        completed, the virtual interface will be in 'Confirming'
        state, and will not be available for handling traffic.

        :type connection_id: string
        :param connection_id: The connection ID on which the public virtual
            interface is provisioned.
        Default: None

        :type owner_account: string
        :param owner_account: The AWS account that will own the new public
            virtual interface.
        Default: None

        :type new_public_virtual_interface_allocation: dict
        :param new_public_virtual_interface_allocation: Detailed information
            for the public virtual interface to be provisioned.
        Default: None

        """
    params = {'connectionId': connection_id, 'ownerAccount': owner_account, 'newPublicVirtualInterfaceAllocation': new_public_virtual_interface_allocation}
    return self.make_request(action='AllocatePublicVirtualInterface', body=json.dumps(params))