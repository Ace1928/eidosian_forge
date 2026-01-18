from . import exceptions
from . import misc
from . import normalizers
def forbid_use_of_password(self):
    """Prevent passwords from being included in the URI.

        .. versionadded:: 1.0

        :returns:
            The validator instance.
        :rtype:
            Validator
        """
    self.allow_password = False
    return self