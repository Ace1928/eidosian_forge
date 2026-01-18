from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
def generateWWWAuthenticate(scheme, challenge):
    lst = []
    for k, v in challenge.items():
        k = ensureBytes(k)
        v = ensureBytes(v)
        lst.append(k + b'=' + quoteString(v))
    return b' '.join([scheme, b', '.join(lst)])