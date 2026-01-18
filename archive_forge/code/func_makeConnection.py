from zope.interface import Attribute, Interface
def makeConnection(xs):
    """
        A connection over the underlying transport of the XML stream has been
        established.

        At this point, no traffic has been exchanged over the XML stream
        given in C{xs}.

        This should setup L{xmlstream} and call L{connectionMade}.

        @type xs:
               L{twisted.words.protocols.jabber.xmlstream.XmlStream}
        """