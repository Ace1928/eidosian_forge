import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def get_public_urls(self, service_type=None, name=None):
    """
        Retrieve all the available public (external) URLs for the provided
        service type and name.
        """
    endpoints = self.get_endpoints(service_type=service_type, name=name)
    result = []
    for endpoint in endpoints:
        endpoint_type = endpoint.endpoint_type
        if endpoint_type == OpenStackIdentityEndpointType.EXTERNAL:
            result.append(endpoint.url)
    return result