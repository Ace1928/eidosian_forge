import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def describe_domain(self, name):
    """
        Returns information about the specified domain including
        description and status.

        :type name: string
        :param name: The name of the domain to describe.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('DescribeDomain', {'name': name})