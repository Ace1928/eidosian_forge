from pprint import pformat
from six import iteritems
import re
@medium.setter
def medium(self, medium):
    """
        Sets the medium of this V1EmptyDirVolumeSource.
        What type of storage medium should back this directory. The default is
        "" which means to use the node's default medium. Must be an empty
        string (default) or Memory. More info:
        https://kubernetes.io/docs/concepts/storage/volumes#emptydir

        :param medium: The medium of this V1EmptyDirVolumeSource.
        :type: str
        """
    self._medium = medium