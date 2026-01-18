from twisted.python import reflect
def removeEntity(self, name, request):
    """Remove an entity for 'name', based on the content of 'request'."""
    raise NotSupportedError('%s.removeEntity' % reflect.qual(self.__class__))