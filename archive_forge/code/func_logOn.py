from zope.interface import Attribute, Interface
def logOn(chatui):
    """
        Go on-line.

        @type chatui: Implementor of C{IChatUI}

        @rtype: L{Deferred} with an eventual L{IClient} result.
        """