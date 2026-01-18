from twisted.python import reflect
def getStaticEntity(self, name):
    """Get an entity that was added to me using putEntity.

        This method will return 'None' if it fails.
        """
    return self.entities.get(name)