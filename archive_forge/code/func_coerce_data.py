import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def coerce_data(data, encoding):
    """Ensure that every object's __len__ behaves uniformly."""
    if not isinstance(data, CustomBytesIO):
        if hasattr(data, 'getvalue'):
            return CustomBytesIO(data.getvalue(), encoding)
        if hasattr(data, 'fileno'):
            return FileWrapper(data)
        if not hasattr(data, 'read'):
            return CustomBytesIO(data, encoding)
    return data