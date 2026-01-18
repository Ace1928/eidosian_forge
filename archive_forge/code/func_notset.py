def notset(self, name):
    """
        Get whether a property has never been set by I{name}.
        @param name: A property name.
        @type name: str
        @return: True if never been set.
        @rtype: bool
        """
    self.provider(name).__notset(name)