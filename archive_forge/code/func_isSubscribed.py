from zope.interface import Interface
def isSubscribed(name):
    """
        Check the subscription status of a mailbox

        @type name: L{bytes}
        @param name: The name of the mailbox to check

        @rtype: L{Deferred} or L{bool}
        @return: A true value if the given mailbox is currently subscribed to,
            a false value otherwise. A L{Deferred} may also be returned whose
            callback will be invoked with one of these values.
        """