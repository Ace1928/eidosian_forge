from twisted.python import reflect
def listEntities(self, request):
    """Retrieve a list of all name, entity pairs I contain.

        See getEntity.
        """
    return self.listStaticEntities() + self.listDynamicEntities(request)