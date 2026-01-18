import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def quilt_pop(working_dir, patch, patches_dir=None, series_file=None, quiet=None):
    """Pop a patch.

    :param working_dir: Directory to work in
    :param patch: Patch to apply
    :param patches_dir: Optional patches directory
    :param series_file: Optional series file
    """
    return run_quilt(['pop', patch], working_dir=working_dir, patches_dir=patches_dir, series_file=series_file, quiet=quiet)