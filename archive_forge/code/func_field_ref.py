from pprint import pformat
from six import iteritems
import re
@field_ref.setter
def field_ref(self, field_ref):
    """
        Sets the field_ref of this V1DownwardAPIVolumeFile.
        Required: Selects a field of the pod: only annotations, labels, name and
        namespace are supported.

        :param field_ref: The field_ref of this V1DownwardAPIVolumeFile.
        :type: V1ObjectFieldSelector
        """
    self._field_ref = field_ref