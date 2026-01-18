import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_destroy_sslcertificate(self, sslcertificate):
    """
        Deletes the specified SslCertificate resource.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  sslcertificate:  Name of the SslCertificate resource to
                                 delete.
        :type   sslcertificate: ``str``

        :return  sslCertificate:  Return True if successful.
        :rtype   sslCertificate: ````bool````
        """
    request = '/global/sslCertificates/%s' % sslcertificate.name
    request_data = {}
    self.connection.async_request(request, method='DELETE', data=request_data)
    return True