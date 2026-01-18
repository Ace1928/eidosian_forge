from pprint import pformat
from six import iteritems
import re
@target_ref.setter
def target_ref(self, target_ref):
    """
        Sets the target_ref of this V1EndpointAddress.
        Reference to object providing the endpoint.

        :param target_ref: The target_ref of this V1EndpointAddress.
        :type: V1ObjectReference
        """
    self._target_ref = target_ref