import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def readable_data(data, encoding):
    """Coerce the data to an object with a ``read`` method."""
    if hasattr(data, 'read'):
        return data
    return CustomBytesIO(data, encoding)