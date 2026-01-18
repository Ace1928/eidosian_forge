import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_all_group_policies(self, group_name, marker=None, max_items=None):
    """
        List the names of the policies associated with the specified group.

        :type group_name: string
        :param group_name: The name of the group the policy is associated with.

        :type marker: string
        :param marker: Use this only when paginating results and only
            in follow-up request after you've received a response
            where the results are truncated.  Set this to the value of
            the Marker element in the response you just received.

        :type max_items: int
        :param max_items: Use this only when paginating results to indicate
            the maximum number of groups you want in the response.
        """
    params = {'GroupName': group_name}
    if marker:
        params['Marker'] = marker
    if max_items:
        params['MaxItems'] = max_items
    return self.get_response('ListGroupPolicies', params, list_marker='PolicyNames')