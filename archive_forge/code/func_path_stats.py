import _imp
import _io
import sys
import _warnings
import marshal
def path_stats(self, path):
    """Return the metadata for the path."""
    st = _path_stat(path)
    return {'mtime': st.st_mtime, 'size': st.st_size}