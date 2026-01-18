from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
def getChildWithDefault(self, name, request):
    """
                Pass through the lookup to the wrapped resource, wrapping
                the result in L{ResourceWrapper} to ensure C{logout} is
                called when rendering of the child is complete.
                """
    return ResourceWrapper(self.resource.getChildWithDefault(name, request))