def getOptionalRelease(self):
    """Return first release in which this feature was recognized.

        This is a 5-tuple, of the same form as sys.version_info.
        """
    return self.optional