from pprint import pformat
from six import iteritems
import re
@empty_dir.setter
def empty_dir(self, empty_dir):
    """
        Sets the empty_dir of this V1Volume.
        EmptyDir represents a temporary directory that shares a pod's lifetime.
        More info: https://kubernetes.io/docs/concepts/storage/volumes#emptydir

        :param empty_dir: The empty_dir of this V1Volume.
        :type: V1EmptyDirVolumeSource
        """
    self._empty_dir = empty_dir