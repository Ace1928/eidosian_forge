from pprint import pformat
from six import iteritems
import re
@storageos.setter
def storageos(self, storageos):
    """
        Sets the storageos of this V1PersistentVolumeSpec.
        StorageOS represents a StorageOS volume that is attached to the
        kubelet's host machine and mounted into the pod More info:
        https://releases.k8s.io/HEAD/examples/volumes/storageos/README.md

        :param storageos: The storageos of this V1PersistentVolumeSpec.
        :type: V1StorageOSPersistentVolumeSource
        """
    self._storageos = storageos