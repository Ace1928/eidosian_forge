from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def buildResolver(res):
    return Resolver(hints=[e[1] for e in res if e[0]], resolverFactory=resolverFactory)