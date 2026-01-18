from pprint import pformat
from six import iteritems
import re
@scale_io.setter
def scale_io(self, scale_io):
    """
        Sets the scale_io of this V1PersistentVolumeSpec.
        ScaleIO represents a ScaleIO persistent volume attached and mounted on
        Kubernetes nodes.

        :param scale_io: The scale_io of this V1PersistentVolumeSpec.
        :type: V1ScaleIOPersistentVolumeSource
        """
    self._scale_io = scale_io