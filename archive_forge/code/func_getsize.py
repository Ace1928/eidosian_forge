import os
import stat
def getsize(filename):
    """Return the size of a file, reported by os.stat()."""
    return os.stat(filename).st_size