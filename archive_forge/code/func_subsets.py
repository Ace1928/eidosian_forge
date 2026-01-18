from pprint import pformat
from six import iteritems
import re
@subsets.setter
def subsets(self, subsets):
    """
        Sets the subsets of this V1Endpoints.
        The set of all endpoints is the union of all subsets. Addresses are
        placed into subsets according to the IPs they share. A single address
        with multiple ports, some of which are ready and some of which are not
        (because they come from different containers) will result in the address
        being displayed in different subsets for the different ports. No address
        will appear in both Addresses and NotReadyAddresses in the same subset.
        Sets of addresses and ports that comprise a service.

        :param subsets: The subsets of this V1Endpoints.
        :type: list[V1EndpointSubset]
        """
    self._subsets = subsets