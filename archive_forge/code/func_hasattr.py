def hasattr(self, key):
    """hasattr function available as a method.

        Implemented like has_key.

        Examples
        --------
        >>> s = Struct(a=10)
        >>> s.hasattr('a')
        True
        >>> s.hasattr('b')
        False
        >>> s.hasattr('get')
        False
        """
    return key in self