from pprint import pformat
from six import iteritems
import re
@claim_ref.setter
def claim_ref(self, claim_ref):
    """
        Sets the claim_ref of this V1PersistentVolumeSpec.
        ClaimRef is part of a bi-directional binding between PersistentVolume
        and PersistentVolumeClaim. Expected to be non-nil when bound.
        claim.VolumeName is the authoritative bind between PV and PVC. More
        info:
        https://kubernetes.io/docs/concepts/storage/persistent-volumes#binding

        :param claim_ref: The claim_ref of this V1PersistentVolumeSpec.
        :type: V1ObjectReference
        """
    self._claim_ref = claim_ref