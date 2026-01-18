import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def describe_applications(self, application_names=None):
    """Returns the descriptions of existing applications.

        :type application_names: list
        :param application_names: If specified, AWS Elastic Beanstalk restricts
            the returned descriptions to only include those with the specified
            names.

        """
    params = {}
    if application_names:
        self.build_list_params(params, application_names, 'ApplicationNames.member')
    return self._get_response('DescribeApplications', params)