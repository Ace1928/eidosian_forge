import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def delete_app(self, app_id):
    """
        Deletes a specified app.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type app_id: string
        :param app_id: The app ID.

        """
    params = {'AppId': app_id}
    return self.make_request(action='DeleteApp', body=json.dumps(params))