def allow_new_attr(self, allow=True):
    """Set whether new attributes can be created in this Struct.

        This can be used to catch typos by verifying that the attribute user
        tries to change already exists in this Struct.
        """
    object.__setattr__(self, '_allownew', allow)