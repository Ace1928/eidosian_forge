import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
@property
def camelcase_name(self):
    """Camelcase name of this field.

    Returns:
      str: the name in CamelCase.
    """
    if self._camelcase_name is None:
        self._camelcase_name = _ToCamelCase(self.name)
    return self._camelcase_name