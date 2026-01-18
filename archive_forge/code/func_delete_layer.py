import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def delete_layer(self, layer_id):
    """
        Deletes a specified layer. You must first stop and then delete
        all associated instances or unassign registered instances. For
        more information, see `How to Delete a Layer`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type layer_id: string
        :param layer_id: The layer ID.

        """
    params = {'LayerId': layer_id}
    return self.make_request(action='DeleteLayer', body=json.dumps(params))