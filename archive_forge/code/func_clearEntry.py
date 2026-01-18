from twisted.internet import defer
from twisted.names import common, dns
from twisted.python import failure, log
def clearEntry(self, query):
    del self.cache[query]
    del self.cancel[query]