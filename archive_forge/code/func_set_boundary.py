import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def set_boundary(self, boundary):
    """Define the boundary used in a multi parts message.

        The file should be at the beginning of the body, the first range
        definition is read and taken into account.
        """
    if not isinstance(boundary, bytes):
        raise TypeError(boundary)
    self._boundary = boundary
    self.read_boundary()
    self.read_range_definition()