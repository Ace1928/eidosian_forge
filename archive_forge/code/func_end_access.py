import sys
def end_access(self):
    """Call this method once you are done using the instance. It is automatically
        called on destruction, and should be called just in time to allow system
        resources to be freed.

        Once you called end_access, you must call begin access before reusing this instance!"""
    self._size = 0
    if self._c is not None:
        self._c.unuse_region()