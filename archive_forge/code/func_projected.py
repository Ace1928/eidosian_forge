from pprint import pformat
from six import iteritems
import re
@projected.setter
def projected(self, projected):
    """
        Sets the projected of this V1Volume.
        Items for all in one resources secrets, configmaps, and downward API

        :param projected: The projected of this V1Volume.
        :type: V1ProjectedVolumeSource
        """
    self._projected = projected