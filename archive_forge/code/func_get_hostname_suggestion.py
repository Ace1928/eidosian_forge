import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def get_hostname_suggestion(self, layer_id):
    """
        Gets a generated host name for the specified layer, based on
        the current host name theme.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type layer_id: string
        :param layer_id: The layer ID.

        """
    params = {'LayerId': layer_id}
    return self.make_request(action='GetHostnameSuggestion', body=json.dumps(params))