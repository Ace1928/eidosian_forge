import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def list_tags_for_resource(self, resource_name):
    """
        Lists all tags on an Amazon RDS resource.

        For an overview on tagging an Amazon RDS resource, see
        `Tagging Amazon RDS Resources`_.

        :type resource_name: string
        :param resource_name: The Amazon RDS resource with tags to be listed.
            This value is an Amazon Resource Name (ARN). For information about
            creating an ARN, see ` Constructing an RDS Amazon Resource Name
            (ARN)`_.

        """
    params = {'ResourceName': resource_name}
    return self._make_request(action='ListTagsForResource', verb='POST', path='/', params=params)