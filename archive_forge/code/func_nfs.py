from pprint import pformat
from six import iteritems
import re
@nfs.setter
def nfs(self, nfs):
    """
        Sets the nfs of this V1PersistentVolumeSpec.
        NFS represents an NFS mount on the host. Provisioned by an admin. More
        info: https://kubernetes.io/docs/concepts/storage/volumes#nfs

        :param nfs: The nfs of this V1PersistentVolumeSpec.
        :type: V1NFSVolumeSource
        """
    self._nfs = nfs