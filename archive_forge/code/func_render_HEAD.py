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
def render_HEAD(self, request):
    """
        Default handling of HEAD method.

        I just return self.render_GET(request). When method is HEAD,
        the framework will handle this correctly.
        """
    return self.render_GET(request)