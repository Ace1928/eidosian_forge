from typing import Optional, Union
from twisted.internet import interfaces
from twisted.internet.endpoints import _WrapperServerEndpoint
from twisted.protocols import policies
from . import _info
from ._exceptions import InvalidProxyHeader
from ._v1parser import V1Parser
from ._v2parser import V2Parser
def logPrefix(self) -> str:
    """
        Annotate the wrapped factory's log prefix with some text indicating
        the PROXY protocol is in use.

        @rtype: C{str}
        """
    if interfaces.ILoggingContext.providedBy(self.wrappedFactory):
        logPrefix = self.wrappedFactory.logPrefix()
    else:
        logPrefix = self.wrappedFactory.__class__.__name__
    return f'{logPrefix} (PROXY)'