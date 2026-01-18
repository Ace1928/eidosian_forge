import importlib
import json
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from os.path import basename, normpath
from os.path import join as pjoin
from jupyter_core.paths import ENV_JUPYTER_PATH, SYSTEM_JUPYTER_PATH, jupyter_data_dir
from jupyter_core.utils import ensure_dir_exists
from jupyter_server.extension.serverextension import ArgumentConflict
from jupyterlab_server.config import get_federated_extensions
from .commands import _test_overlap
def develop_labextension(path, symlink=True, overwrite=False, user=False, labextensions_dir=None, destination=None, logger=None, sys_prefix=False):
    """Install a prebuilt extension for JupyterLab

    Stages files and/or directories into the labextensions directory.
    By default, this compares modification time, and only stages files that need updating.
    If `overwrite` is specified, matching files are purged before proceeding.

    Parameters
    ----------

    path : path to file, directory, zip or tarball archive, or URL to install
        By default, the file will be installed with its base name, so '/path/to/foo'
        will install to 'labextensions/foo'. See the destination argument below to change this.
        Archives (zip or tarballs) will be extracted into the labextensions directory.
    user : bool [default: False]
        Whether to install to the user's labextensions directory.
        Otherwise do a system-wide install (e.g. /usr/local/share/jupyter/labextensions).
    overwrite : bool [default: False]
        If True, always install the files, regardless of what may already be installed.
    symlink : bool [default: True]
        If True, create a symlink in labextensions, rather than copying files.
        Windows support for symlinks requires a permission bit which only admin users
        have by default, so don't rely on it.
    labextensions_dir : str [optional]
        Specify absolute path of labextensions directory explicitly.
    destination : str [optional]
        name the labextension is installed to.  For example, if destination is 'foo', then
        the source file will be installed to 'labextensions/foo', regardless of the source name.
    logger : Jupyter logger [optional]
        Logger instance to use
    """
    full_dest = None
    labext = _get_labextension_dir(user=user, sys_prefix=sys_prefix, labextensions_dir=labextensions_dir)
    ensure_dir_exists(labext)
    if isinstance(path, (list, tuple)):
        msg = 'path must be a string pointing to a single extension to install; call this function multiple times to install multiple extensions'
        raise TypeError(msg)
    if not destination:
        destination = basename(normpath(path))
    full_dest = normpath(pjoin(labext, destination))
    if overwrite and os.path.lexists(full_dest):
        if logger:
            logger.info('Removing: %s' % full_dest)
        if os.path.isdir(full_dest) and (not os.path.islink(full_dest)):
            shutil.rmtree(full_dest)
        else:
            os.remove(full_dest)
    os.makedirs(os.path.dirname(full_dest), exist_ok=True)
    if symlink:
        path = os.path.abspath(path)
        if not os.path.exists(full_dest):
            if logger:
                logger.info(f'Symlinking: {full_dest} -> {path}')
            try:
                os.symlink(path, full_dest)
            except OSError as e:
                if platform.platform().startswith('Windows'):
                    msg = "Symlinks can be activated on Windows 10 for Python version 3.8 or higher by activating the 'Developer Mode'. That may not be allowed by your administrators.\nSee https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development"
                    raise OSError(msg) from e
                raise
        elif not os.path.islink(full_dest):
            raise ValueError('%s exists and is not a symlink' % full_dest)
    elif os.path.isdir(path):
        path = pjoin(os.path.abspath(path), '')
        for parent, _, files in os.walk(path):
            dest_dir = pjoin(full_dest, parent[len(path):])
            if not os.path.exists(dest_dir):
                if logger:
                    logger.info('Making directory: %s' % dest_dir)
                os.makedirs(dest_dir)
            for file_name in files:
                src = pjoin(parent, file_name)
                dest_file = pjoin(dest_dir, file_name)
                _maybe_copy(src, dest_file, logger=logger)
    else:
        src = path
        _maybe_copy(src, full_dest, logger=logger)
    return full_dest