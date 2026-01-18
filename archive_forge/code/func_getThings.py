def getThings(self):
    """Returns the set of all things known so far.

        The return value belongs to the Classifier object and should NOT
        be modified while the classifier is still in use.
        """
    self._process()
    return self._things