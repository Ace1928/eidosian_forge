from twisted.python import reflect
def listNames(self, request):
    """Retrieve a list of all names for entities that I contain.

        See getEntity.
        """
    return self.listStaticNames()