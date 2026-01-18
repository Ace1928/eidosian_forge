import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def quilt_push(working_dir, patch, patches_dir=None, series_file=None, quiet=None, force=False, refresh=False):
    """Push a patch.

    :param working_dir: Directory to work in
    :param patch: Patch to push
    :param patches_dir: Optional patches directory
    :param series_file: Optional series file
    :param force: Force push
    :param refresh: Refresh
    """
    args = []
    if force:
        args.append('-f')
    if refresh:
        args.append('--refresh')
    return run_quilt(['push', patch] + args, working_dir=working_dir, patches_dir=patches_dir, series_file=series_file, quiet=quiet)