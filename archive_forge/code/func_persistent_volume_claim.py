from pprint import pformat
from six import iteritems
import re
@persistent_volume_claim.setter
def persistent_volume_claim(self, persistent_volume_claim):
    """
        Sets the persistent_volume_claim of this V1Volume.
        PersistentVolumeClaimVolumeSource represents a reference to a
        PersistentVolumeClaim in the same namespace. More info:
        https://kubernetes.io/docs/concepts/storage/persistent-volumes#persistentvolumeclaims

        :param persistent_volume_claim: The persistent_volume_claim of this
        V1Volume.
        :type: V1PersistentVolumeClaimVolumeSource
        """
    self._persistent_volume_claim = persistent_volume_claim