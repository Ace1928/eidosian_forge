import re
def ensureNoSlash(self):
    """Return a URI with the path normalised to not end with a slash."""
    if self.path.endswith('/'):
        return self.replace(path=self.path.rstrip('/'))
    else:
        return self