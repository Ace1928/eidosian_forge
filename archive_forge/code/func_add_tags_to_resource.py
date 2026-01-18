import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def add_tags_to_resource(self, resource_name, tags):
    """
        Adds metadata tags to an Amazon RDS resource. These tags can
        also be used with cost allocation reporting to track cost
        associated with Amazon RDS resources, or used in Condition
        statement in IAM policy for Amazon RDS.

        For an overview on tagging Amazon RDS resources, see `Tagging
        Amazon RDS Resources`_.

        :type resource_name: string
        :param resource_name: The Amazon RDS resource the tags will be added
            to. This value is an Amazon Resource Name (ARN). For information
            about creating an ARN, see ` Constructing an RDS Amazon Resource
            Name (ARN)`_.

        :type tags: list
        :param tags: The tags to be assigned to the Amazon RDS resource.
            Tags must be passed as tuples in the form
            [('key1', 'valueForKey1'), ('key2', 'valueForKey2')]

        """
    params = {'ResourceName': resource_name}
    self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='AddTagsToResource', verb='POST', path='/', params=params)