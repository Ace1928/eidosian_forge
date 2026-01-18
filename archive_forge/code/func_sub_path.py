from pprint import pformat
from six import iteritems
import re
@sub_path.setter
def sub_path(self, sub_path):
    """
        Sets the sub_path of this V1VolumeMount.
        Path within the volume from which the container's volume should be
        mounted. Defaults to "" (volume's root).

        :param sub_path: The sub_path of this V1VolumeMount.
        :type: str
        """
    self._sub_path = sub_path