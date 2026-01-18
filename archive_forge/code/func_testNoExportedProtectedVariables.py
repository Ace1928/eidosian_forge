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
def testNoExportedProtectedVariables(self):
    """Test that there are no protected variables listed in __all__."""
    protected_variables = []
    for attribute in self.MODULE.__all__:
        if attribute.startswith('_'):
            protected_variables.append(attribute)
    if protected_variables:
        self.fail('%s are protected variables and may not be exported.' % protected_variables)