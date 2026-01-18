from zope.interface import Attribute, Interface
def createGroup(name):
    """Create a new group with the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the created group, or
        with fails with L{twisted.words.ewords.DuplicateGroup} if a
        group by that name exists already.
        """