from pprint import pformat
from six import iteritems
import re
@deletion_grace_period_seconds.setter
def deletion_grace_period_seconds(self, deletion_grace_period_seconds):
    """
        Sets the deletion_grace_period_seconds of this V1ObjectMeta.
        Number of seconds allowed for this object to gracefully terminate before
        it will be removed from the system. Only set when deletionTimestamp is
        also set. May only be shortened. Read-only.

        :param deletion_grace_period_seconds: The deletion_grace_period_seconds
        of this V1ObjectMeta.
        :type: int
        """
    self._deletion_grace_period_seconds = deletion_grace_period_seconds