import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def list_available_zones(self):
    """
        Lists the Availability Zones that have available AWS CloudHSM
        capacity.

        
        """
    params = {}
    return self.make_request(action='ListAvailableZones', body=json.dumps(params))