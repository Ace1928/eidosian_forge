import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
@availability_condition.setter
def availability_condition(self, value):
    """Updates the current availability condition.

        Args:
            value (Optional[google.auth.downscoped.AvailabilityCondition]): The updated
                value of the availability condition.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type google.auth.downscoped.AvailabilityCondition
                or None.
        """
    if not isinstance(value, AvailabilityCondition) and value is not None:
        raise exceptions.InvalidType("The provided availability_condition is not a 'google.auth.downscoped.AvailabilityCondition' or None.")
    self._availability_condition = value