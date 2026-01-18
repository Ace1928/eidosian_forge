import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def quilt_series(tree, series_path):
    """Find the list of patches.

    :param tree: Tree to read from
    """
    try:
        return [os.fsdecode(patch.rstrip(b'\n')) for patch in tree.get_file_lines(series_path) if patch.strip() != b'']
    except OSError as e:
        if e.errno == errno.ENOENT:
            return []
        raise
    except _mod_transport.NoSuchFile:
        return []