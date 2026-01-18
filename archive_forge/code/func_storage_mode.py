from pprint import pformat
from six import iteritems
import re
@storage_mode.setter
def storage_mode(self, storage_mode):
    """
        Sets the storage_mode of this V1ScaleIOVolumeSource.
        Indicates whether the storage for a volume should be ThickProvisioned or
        ThinProvisioned. Default is ThinProvisioned.

        :param storage_mode: The storage_mode of this V1ScaleIOVolumeSource.
        :type: str
        """
    self._storage_mode = storage_mode