import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def start_stack(self, stack_id):
    """
        Starts a stack's instances.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID.

        """
    params = {'StackId': stack_id}
    return self.make_request(action='StartStack', body=json.dumps(params))