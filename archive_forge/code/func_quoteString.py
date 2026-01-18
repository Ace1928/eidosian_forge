from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
def quoteString(s):
    return b'"' + s.replace(b'\\', b'\\\\').replace(b'"', b'\\"') + b'"'