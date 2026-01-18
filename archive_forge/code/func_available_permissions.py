import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
@available_permissions.setter
def available_permissions(self, value):
    """Updates the current available permissions.

        Args:
            value (Sequence[str]): The updated value of the available permissions.

        Raises:
            InvalidType: If the value is not a list of strings.
            InvalidValue: If the value is not valid.
        """
    for available_permission in value:
        if not isinstance(available_permission, six.string_types):
            raise exceptions.InvalidType('Provided available_permissions are not a list of strings.')
        if available_permission.find('inRole:') != 0:
            raise exceptions.InvalidValue("available_permissions must be prefixed with 'inRole:'.")
    self._available_permissions = list(value)