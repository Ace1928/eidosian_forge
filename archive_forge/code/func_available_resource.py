import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
@available_resource.setter
def available_resource(self, value):
    """Updates the current available resource.

        Args:
            value (str): The updated value of the available resource.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not a string.
        """
    if not isinstance(value, six.string_types):
        raise exceptions.InvalidType('The provided available_resource is not a string.')
    self._available_resource = value