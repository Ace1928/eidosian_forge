from . import exceptions
from . import misc
from . import normalizers
def allow_use_of_password(self):
    """Allow passwords to be present in the URI.

        .. versionadded:: 1.0

        :returns:
            The validator instance.
        :rtype:
            Validator
        """
    self.allow_password = True
    return self