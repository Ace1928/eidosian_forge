import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def delete_interconnect(self, interconnect_id):
    """
        Deletes the specified interconnect.

        :type interconnect_id: string
        :param interconnect_id: The ID of the interconnect.
        Example: dxcon-abc123

        """
    params = {'interconnectId': interconnect_id}
    return self.make_request(action='DeleteInterconnect', body=json.dumps(params))