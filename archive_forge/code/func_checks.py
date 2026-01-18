import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
def checks(self, format, raises=()):
    """
        Register a decorated function as validating a new format.

        Arguments:

            format (str):

                The format that the decorated function will check.

            raises (Exception):

                The exception(s) raised by the decorated function when
                an invalid instance is found.

                The exception object will be accessible as the
                :attr:`ValidationError.cause` attribute of the resulting
                validation error.

        """

    def _checks(func):
        self.checkers[format] = (func, raises)
        return func
    return _checks