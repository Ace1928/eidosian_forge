from __future__ import annotations
import warnings
from typing import Sequence
from zope.interface import Attribute, Interface, implementer
from incremental import Version
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import prefixedMethodNames
from twisted.web._responses import FORBIDDEN, NOT_FOUND
from twisted.web.error import UnsupportedMethod
def getEncoder(self, request):
    """
        Browser the list of encoders looking for one applicable encoder.
        """
    for encoderFactory in self._encoders:
        encoder = encoderFactory.encoderForRequest(request)
        if encoder is not None:
            return encoder