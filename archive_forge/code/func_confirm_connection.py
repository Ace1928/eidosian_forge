import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def confirm_connection(self, connection_id):
    """
        Confirm the creation of a hosted connection on an
        interconnect.

        Upon creation, the hosted connection is initially in the
        'Ordering' state, and will remain in this state until the
        owner calls ConfirmConnection to confirm creation of the
        hosted connection.

        :type connection_id: string
        :param connection_id: ID of the connection.
        Example: dxcon-fg5678gh

        Default: None

        """
    params = {'connectionId': connection_id}
    return self.make_request(action='ConfirmConnection', body=json.dumps(params))