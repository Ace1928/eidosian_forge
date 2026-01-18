import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def makeStatefulDispatcher(name, template):
    """
    Given a I{dispatch} name and a function, return a function which can be
    used as a method and which, when called, will call another method defined
    on the instance and return the result.  The other method which is called is
    determined by the value of the C{_state} attribute of the instance.

    @param name: A string which is used to construct the name of the subsidiary
        method to invoke.  The subsidiary method is named like C{'_%s_%s' %
        (name, _state)}.

    @param template: A function object which is used to give the returned
        function a docstring.

    @return: The dispatcher function.
    """

    def dispatcher(self, *args, **kwargs):
        func = getattr(self, '_' + name + '_' + self._state, None)
        if func is None:
            raise RuntimeError(f'{self!r} has no {name} method in state {self._state}')
        return func(*args, **kwargs)
    dispatcher.__doc__ = template.__doc__
    return dispatcher