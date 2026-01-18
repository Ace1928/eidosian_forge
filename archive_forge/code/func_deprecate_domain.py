import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def deprecate_domain(self, name):
    """
        Deprecates the specified domain. After a domain has been
        deprecated it cannot be used to create new workflow executions
        or register new types. However, you can still use visibility
        actions on this domain. Deprecating a domain also deprecates
        all activity and workflow types registered in the
        domain. Executions that were started before the domain was
        deprecated will continue to run.

        :type name: string
        :param name: The name of the domain to deprecate.

        :raises: UnknownResourceFault, DomainDeprecatedFault,
            SWFOperationNotPermittedError
        """
    return self.json_request('DeprecateDomain', {'name': name})