import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def stop_logging(self, name):
    """
        Suspends the recording of AWS API calls and log file delivery
        for the specified trail. Under most circumstances, there is no
        need to use this action. You can update a trail without
        stopping it first. This action is the only way to stop
        recording.

        :type name: string
        :param name: Communicates to CloudTrail the name of the trail for which
            to stop logging AWS API calls.

        """
    params = {'Name': name}
    return self.make_request(action='StopLogging', body=json.dumps(params))