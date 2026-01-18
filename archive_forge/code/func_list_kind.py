from pprint import pformat
from six import iteritems
import re
@list_kind.setter
def list_kind(self, list_kind):
    """
        Sets the list_kind of this V1beta1CustomResourceDefinitionNames.
        ListKind is the serialized kind of the list for this resource.  Defaults
        to <kind>List.

        :param list_kind: The list_kind of this
        V1beta1CustomResourceDefinitionNames.
        :type: str
        """
    self._list_kind = list_kind