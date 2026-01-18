from pprint import pformat
from six import iteritems
import re
@storage_version_hash.setter
def storage_version_hash(self, storage_version_hash):
    """
        Sets the storage_version_hash of this V1APIResource.
        The hash value of the storage version, the version this resource is
        converted to when written to the data store. Value must be treated as
        opaque by clients. Only equality comparison on the value is valid. This
        is an alpha feature and may change or be removed in the future. The
        field is populated by the apiserver only if the StorageVersionHash
        feature gate is enabled. This field will remain optional even if it
        graduates.

        :param storage_version_hash: The storage_version_hash of this
        V1APIResource.
        :type: str
        """
    self._storage_version_hash = storage_version_hash