import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def deprecate_activity_type(self, domain, activity_name, activity_version):
    """
        Deprecates the specified activity type. After an activity 
        type has been deprecated, you cannot create new tasks of 
        that activity type. Tasks of this type that were scheduled 
        before the type was deprecated will continue to run.

        :type domain: string
        :param domain: The name of the domain in which the activity
            type is registered.

        :type activity_name: string
        :param activity_name: The name of this activity.

        :type activity_version: string
        :param activity_version: The version of this activity.

        :raises: UnknownResourceFault, TypeDeprecatedFault,
            SWFOperationNotPermittedError
        """
    return self.json_request('DeprecateActivityType', {'domain': domain, 'activityType': {'name': activity_name, 'version': activity_version}})