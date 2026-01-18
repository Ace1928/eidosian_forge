from pprint import pformat
from six import iteritems
import re
@managed_fields.setter
def managed_fields(self, managed_fields):
    """
        Sets the managed_fields of this V1ObjectMeta.
        ManagedFields maps workflow-id and version to the set of fields that are
        managed by that workflow. This is mostly for internal housekeeping, and
        users typically shouldn't need to set or understand this field. A
        workflow can be the user's name, a controller's name, or the name of a
        specific apply path like "ci-cd". The set of fields is always in the
        version that the workflow used when modifying the object.  This field is
        alpha and can be changed or removed without notice.

        :param managed_fields: The managed_fields of this V1ObjectMeta.
        :type: list[V1ManagedFieldsEntry]
        """
    self._managed_fields = managed_fields