import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def quilt_applied(tree):
    """Find the list of applied quilt patches.

    """
    try:
        return [os.fsdecode(patch.rstrip(b'\n')) for patch in tree.get_file_lines('.pc/applied-patches') if patch.strip() != b'']
    except _mod_transport.NoSuchFile:
        return []
    except OSError as e:
        if e.errno == errno.ENOENT:
            return []
        raise