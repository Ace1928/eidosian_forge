import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def describe_availability_options(self, domain_name, deployed=None):
    """
        Gets the availability options configured for a domain. By
        default, shows the configuration with any pending changes. Set
        the `Deployed` option to `True` to show the active
        configuration and exclude pending changes. For more
        information, see `Configuring Availability Options`_ in the
        Amazon CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: The name of the domain you want to describe.

        :type deployed: boolean
        :param deployed: Whether to display the deployed configuration (
            `True`) or include any pending changes ( `False`). Defaults to
            `False`.

        """
    params = {'DomainName': domain_name}
    if deployed is not None:
        params['Deployed'] = str(deployed).lower()
    return self._make_request(action='DescribeAvailabilityOptions', verb='POST', path='/', params=params)