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
def ex_get_sslcertificate(self, name):
    """
        Returns the specified SslCertificate resource. Get a list of available
        SSL certificates by making a list() request.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :param  name:  Name of the SslCertificate resource to
                                 return.
        :type   name: ``str``

        :return:  `GCESslCertificate` object.
        :rtype: :class:`GCESslCertificate`
        """
    request = '/global/sslCertificates/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_sslcertificate(response)