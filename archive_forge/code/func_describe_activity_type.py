import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def describe_activity_type(self, domain, activity_name, activity_version):
    """
        Returns information about the specified activity type. This
        includes configuration settings provided at registration time
        as well as other general information about the type.

        :type domain: string
        :param domain: The name of the domain in which the activity
            type is registered.

        :type activity_name: string
        :param activity_name: The name of this activity.

        :type activity_version: string
        :param activity_version: The version of this activity.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('DescribeActivityType', {'domain': domain, 'activityType': {'name': activity_name, 'version': activity_version}})