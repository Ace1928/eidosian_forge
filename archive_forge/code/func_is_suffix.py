def is_suffix(self):
    """Returns True if this is a suffix range.

        A suffix range is one that specifies the last N bytes of a
        file regardless of file size.

        """
    return self.first == None