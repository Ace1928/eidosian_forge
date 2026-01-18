from pprint import pformat
from six import iteritems
import re
@subresource.setter
def subresource(self, subresource):
    """
        Sets the subresource of this V1beta1ResourceAttributes.
        Subresource is one of the existing resource types.  "" means none.

        :param subresource: The subresource of this V1beta1ResourceAttributes.
        :type: str
        """
    self._subresource = subresource