import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def remove_user_from_group(self, group_name, user_name):
    """
        Remove a user from a group.

        :type group_name: string
        :param group_name: The name of the group

        :type user_name: string
        :param user_name: The user to remove from the group.

        """
    params = {'GroupName': group_name, 'UserName': user_name}
    return self.get_response('RemoveUserFromGroup', params)