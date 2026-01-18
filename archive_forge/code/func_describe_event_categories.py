import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_event_categories(self, source_type=None):
    """
        Displays a list of event categories for all event source
        types, or for a specified source type. For a list of the event
        categories and source types, go to `Amazon Redshift Event
        Notifications`_.

        :type source_type: string
        :param source_type: The source type, such as cluster or parameter
            group, to which the described event categories apply.
        Valid values: cluster, snapshot, parameter group, and security group.

        """
    params = {}
    if source_type is not None:
        params['SourceType'] = source_type
    return self._make_request(action='DescribeEventCategories', verb='POST', path='/', params=params)