from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def removeRoute(self, destination, xs):
    """
        Remove a route.

        @param destination: Destination of the route that should be removed.
        @type destination: C{str}.
        @param xs: XML Stream to remove the route for.
        @type xs: L{EventDispatcher<utility.EventDispatcher>}.
        """
    xs.removeObserver('/*', self.route)
    if xs == self.routes[destination]:
        del self.routes[destination]