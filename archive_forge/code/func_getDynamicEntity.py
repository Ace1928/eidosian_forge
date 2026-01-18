from twisted.python import reflect
def getDynamicEntity(self, name, request):
    """Subclass this to generate an entity on demand.

        This method should return 'None' if it fails.
        """