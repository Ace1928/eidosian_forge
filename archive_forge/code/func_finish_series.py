from . import commit, controldir, errors, revision
def finish_series(self):
    """Call this after start_series to unlock the various objects."""
    self._tree.unlock()
    self._tree = None