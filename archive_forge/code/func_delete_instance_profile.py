import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_instance_profile(self, instance_profile_name):
    """
        Deletes the specified instance profile. The instance profile must not
        have an associated role.

        :type instance_profile_name: string
        :param instance_profile_name: Name of the instance profile to delete.
        """
    return self.get_response('DeleteInstanceProfile', {'InstanceProfileName': instance_profile_name})