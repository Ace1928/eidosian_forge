from pprint import pformat
from six import iteritems
import re
@quobyte.setter
def quobyte(self, quobyte):
    """
        Sets the quobyte of this V1PersistentVolumeSpec.
        Quobyte represents a Quobyte mount on the host that shares a pod's
        lifetime

        :param quobyte: The quobyte of this V1PersistentVolumeSpec.
        :type: V1QuobyteVolumeSource
        """
    self._quobyte = quobyte