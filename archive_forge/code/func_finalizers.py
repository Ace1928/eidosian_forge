from pprint import pformat
from six import iteritems
import re
@finalizers.setter
def finalizers(self, finalizers):
    """
        Sets the finalizers of this V1ObjectMeta.
        Must be empty before the object is deleted from the registry. Each entry
        is an identifier for the responsible component that will remove the
        entry from the list. If the deletionTimestamp of the object is non-nil,
        entries in this list can only be removed.

        :param finalizers: The finalizers of this V1ObjectMeta.
        :type: list[str]
        """
    self._finalizers = finalizers