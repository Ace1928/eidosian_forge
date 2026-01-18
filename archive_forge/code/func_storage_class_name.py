from pprint import pformat
from six import iteritems
import re
@storage_class_name.setter
def storage_class_name(self, storage_class_name):
    """
        Sets the storage_class_name of this V1PersistentVolumeSpec.
        Name of StorageClass to which this persistent volume belongs. Empty
        value means that this volume does not belong to any StorageClass.

        :param storage_class_name: The storage_class_name of this
        V1PersistentVolumeSpec.
        :type: str
        """
    self._storage_class_name = storage_class_name