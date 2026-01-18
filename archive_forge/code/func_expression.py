import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
@expression.setter
def expression(self, value):
    """Updates the current condition expression.

        Args:
            value (str): The updated value of the condition expression.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string.
        """
    if not isinstance(value, six.string_types):
        raise exceptions.InvalidType('The provided expression is not a string.')
    self._expression = value