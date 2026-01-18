import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def testAllExported(self):
    """Test that all public attributes not imported are in __all__."""
    missing_attributes = []
    for attribute in dir(self.MODULE):
        if not attribute.startswith('_'):
            if attribute not in self.MODULE.__all__ and (not isinstance(getattr(self.MODULE, attribute), types.ModuleType)) and (attribute != 'with_statement'):
                missing_attributes.append(attribute)
    if missing_attributes:
        self.fail('%s are not modules and not defined in __all__.' % missing_attributes)