import re
def ensureSlash(self):
    """Return a URI with the path normalised to end with a slash."""
    if self.path.endswith('/'):
        return self
    else:
        return self.replace(path=self.path + '/')