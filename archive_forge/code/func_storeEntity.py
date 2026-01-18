from twisted.python import reflect
def storeEntity(self, name, request):
    """Store an entity for 'name', based on the content of 'request'."""
    raise NotSupportedError('%s.storeEntity' % reflect.qual(self.__class__))