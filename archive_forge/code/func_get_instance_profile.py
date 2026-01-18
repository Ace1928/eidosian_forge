import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_instance_profile(self, instance_profile_name):
    """
        Retrieves information about the specified instance profile, including
        the instance profile's path, GUID, ARN, and role.

        :type instance_profile_name: string
        :param instance_profile_name: Name of the instance profile to get
            information about.
        """
    return self.get_response('GetInstanceProfile', {'InstanceProfileName': instance_profile_name})