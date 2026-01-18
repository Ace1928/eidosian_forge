from pprint import pformat
from six import iteritems
import re
@flocker.setter
def flocker(self, flocker):
    """
        Sets the flocker of this V1PersistentVolumeSpec.
        Flocker represents a Flocker volume attached to a kubelet's host machine
        and exposed to the pod for its usage. This depends on the Flocker
        control service being running

        :param flocker: The flocker of this V1PersistentVolumeSpec.
        :type: V1FlockerVolumeSource
        """
    self._flocker = flocker