from pprint import pformat
from six import iteritems
import re
@size_limit.setter
def size_limit(self, size_limit):
    """
        Sets the size_limit of this V1EmptyDirVolumeSource.
        Total amount of local storage required for this EmptyDir volume. The
        size limit is also applicable for memory medium. The maximum usage on
        memory medium EmptyDir would be the minimum value between the SizeLimit
        specified here and the sum of memory limits of all containers in a pod.
        The default is nil which means that the limit is undefined. More info:
        http://kubernetes.io/docs/user-guide/volumes#emptydir

        :param size_limit: The size_limit of this V1EmptyDirVolumeSource.
        :type: str
        """
    self._size_limit = size_limit