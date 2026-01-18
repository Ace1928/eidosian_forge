from pprint import pformat
from six import iteritems
import re
@phase.setter
def phase(self, phase):
    """
        Sets the phase of this V1PersistentVolumeClaimStatus.
        Phase represents the current phase of PersistentVolumeClaim.

        :param phase: The phase of this V1PersistentVolumeClaimStatus.
        :type: str
        """
    self._phase = phase