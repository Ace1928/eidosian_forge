from pprint import pformat
from six import iteritems
import re
@owner_references.setter
def owner_references(self, owner_references):
    """
        Sets the owner_references of this V1ObjectMeta.
        List of objects depended by this object. If ALL objects in the list have
        been deleted, this object will be garbage collected. If this object is
        managed by a controller, then an entry in this list will point to this
        controller, with the controller field set to true. There cannot be more
        than one managing controller.

        :param owner_references: The owner_references of this V1ObjectMeta.
        :type: list[V1OwnerReference]
        """
    self._owner_references = owner_references