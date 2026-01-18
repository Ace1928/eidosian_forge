import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_role(self, role_name):
    """
        Deletes the specified role. The role must not have any policies
        attached.

        :type role_name: string
        :param role_name: Name of the role to delete.
        """
    return self.get_response('DeleteRole', {'RoleName': role_name})