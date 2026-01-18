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
def getChildForRequest(self, request):
    """
        Deprecated in favor of L{getChildForRequest}.

        @see: L{twisted.web.resource.getChildForRequest}.
        """
    warnings.warn('Please use module level getChildForRequest.', DeprecationWarning, 2)
    return getChildForRequest(self, request)