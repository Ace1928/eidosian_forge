import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def quilt_delete(working_dir, patch, patches_dir=None, series_file=None, remove=False):
    """Delete a patch.

    :param working_dir: Directory to work in
    :param patch: Patch to push
    :param patches_dir: Optional patches directory
    :param series_file: Optional series file
    :param remove: Remove the patch file as well
    """
    args = []
    if remove:
        args.append('-r')
    return run_quilt(['delete', patch] + args, working_dir=working_dir, patches_dir=patches_dir, series_file=series_file)