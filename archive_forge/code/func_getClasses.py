def getClasses(self):
    """Returns the list of class sets.

        The return value belongs to the Classifier object and should NOT
        be modified while the classifier is still in use.
        """
    self._process()
    return self._sets