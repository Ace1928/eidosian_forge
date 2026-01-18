from pprint import pformat
from six import iteritems
import re
@publish_not_ready_addresses.setter
def publish_not_ready_addresses(self, publish_not_ready_addresses):
    """
        Sets the publish_not_ready_addresses of this V1ServiceSpec.
        publishNotReadyAddresses, when set to true, indicates that DNS
        implementations must publish the notReadyAddresses of subsets for the
        Endpoints associated with the Service. The default value is false. The
        primary use case for setting this field is to use a StatefulSet's
        Headless Service to propagate SRV records for its Pods without respect
        to their readiness for purpose of peer discovery.

        :param publish_not_ready_addresses: The publish_not_ready_addresses of
        this V1ServiceSpec.
        :type: bool
        """
    self._publish_not_ready_addresses = publish_not_ready_addresses