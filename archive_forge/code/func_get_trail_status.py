import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def get_trail_status(self, name):
    """
        Returns a JSON-formatted list of information about the
        specified trail. Fields include information on delivery
        errors, Amazon SNS and Amazon S3 errors, and start and stop
        logging times for each trail.

        :type name: string
        :param name: The name of the trail for which you are requesting the
            current status.

        """
    params = {'Name': name}
    return self.make_request(action='GetTrailStatus', body=json.dumps(params))