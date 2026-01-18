from pprint import pformat
from six import iteritems
import re
@claim_name.setter
def claim_name(self, claim_name):
    """
        Sets the claim_name of this V1PersistentVolumeClaimVolumeSource.
        ClaimName is the name of a PersistentVolumeClaim in the same namespace
        as the pod using this volume. More info:
        https://kubernetes.io/docs/concepts/storage/persistent-volumes#persistentvolumeclaims

        :param claim_name: The claim_name of this
        V1PersistentVolumeClaimVolumeSource.
        :type: str
        """
    if claim_name is None:
        raise ValueError('Invalid value for `claim_name`, must not be `None`')
    self._claim_name = claim_name