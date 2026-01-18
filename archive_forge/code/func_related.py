from pprint import pformat
from six import iteritems
import re
@related.setter
def related(self, related):
    """
        Sets the related of this V1beta1Event.
        Optional secondary object for more complex actions. E.g. when regarding
        object triggers a creation or deletion of related object.

        :param related: The related of this V1beta1Event.
        :type: V1ObjectReference
        """
    self._related = related