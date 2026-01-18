from pprint import pformat
from six import iteritems
import re
@not_ready_addresses.setter
def not_ready_addresses(self, not_ready_addresses):
    """
        Sets the not_ready_addresses of this V1EndpointSubset.
        IP addresses which offer the related ports but are not currently marked
        as ready because they have not yet finished starting, have recently
        failed a readiness check, or have recently failed a liveness check.

        :param not_ready_addresses: The not_ready_addresses of this
        V1EndpointSubset.
        :type: list[V1EndpointAddress]
        """
    self._not_ready_addresses = not_ready_addresses