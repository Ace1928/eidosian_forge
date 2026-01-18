from pprint import pformat
from six import iteritems
import re
@flex_volume.setter
def flex_volume(self, flex_volume):
    """
        Sets the flex_volume of this V1PersistentVolumeSpec.
        FlexVolume represents a generic volume resource that is
        provisioned/attached using an exec based plugin.

        :param flex_volume: The flex_volume of this V1PersistentVolumeSpec.
        :type: V1FlexPersistentVolumeSource
        """
    self._flex_volume = flex_volume