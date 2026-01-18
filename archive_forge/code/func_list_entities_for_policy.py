import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def list_entities_for_policy(self, policy_arn, path_prefix=None, marker=None, max_items=None, entity_filter=None):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to get entities for

        :type marker: string
        :param marker: A marker used for pagination (received from previous
            accesses)

        :type max_items: int
        :param max_items: Send only max_items; allows paginations

        :type path_prefix: string
        :param path_prefix: Send only items prefixed by this path

        :type entity_filter: string
        :param entity_filter: Which entity type of User | Role | Group |
            LocalManagedPolicy | AWSManagedPolicy to return

        """
    params = {'PolicyArn': policy_arn}
    if marker is not None:
        params['Marker'] = marker
    if max_items is not None:
        params['MaxItems'] = max_items
    if path_prefix is not None:
        params['PathPrefix'] = path_prefix
    if entity_filter is not None:
        params['EntityFilter'] = entity_filter
    return self.get_response('ListEntitiesForPolicy', params, list_marker=('PolicyGroups', 'PolicyUsers', 'PolicyRoles'))