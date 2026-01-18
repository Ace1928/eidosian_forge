from zope.interface import Interface
def listMailboxes(ref, wildcard):
    """
        List all the mailboxes that meet a certain criteria

        @type ref: L{bytes}
        @param ref: The context in which to apply the wildcard

        @type wildcard: L{bytes}
        @param wildcard: An expression against which to match mailbox names.
            '*' matches any number of characters in a mailbox name, and '%'
            matches similarly, but will not match across hierarchical
            boundaries.

        @rtype: L{list} of L{tuple}
        @return: A list of C{(mailboxName, mailboxObject)} which meet the given
            criteria. C{mailboxObject} should implement either
            C{IMailboxIMAPInfo} or C{IMailboxIMAP}. A Deferred may also be
            returned.
        """