import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def describe_scaling_parameters(self, domain_name):
    """
        Gets the scaling parameters configured for a domain. A
        domain's scaling parameters specify the desired search
        instance type and replication count. For more information, see
        `Configuring Scaling Options`_ in the Amazon CloudSearch
        Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        """
    params = {'DomainName': domain_name}
    return self._make_request(action='DescribeScalingParameters', verb='POST', path='/', params=params)