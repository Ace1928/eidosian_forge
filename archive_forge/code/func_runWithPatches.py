def runWithPatches(self, f, *args, **kw):
    """
        Apply each patch already specified. Then run the function f with the
        given args and kwargs. Restore everything when done.
        """
    self.patch()
    try:
        return f(*args, **kw)
    finally:
        self.restore()