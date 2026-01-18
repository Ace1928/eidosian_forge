from pprint import pformat
from six import iteritems
import re
@csi.setter
def csi(self, csi):
    """
        Sets the csi of this V1PersistentVolumeSpec.
        CSI represents storage that is handled by an external CSI driver (Beta
        feature).

        :param csi: The csi of this V1PersistentVolumeSpec.
        :type: V1CSIPersistentVolumeSource
        """
    self._csi = csi