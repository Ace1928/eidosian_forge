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
def develop_labextension_py(module, user=False, sys_prefix=False, overwrite=True, symlink=True, labextensions_dir=None, logger=None):
    """Develop a labextension bundled in a Python package.

    Returns a list of installed/updated directories.

    See develop_labextension for parameter information."""
    m, labexts = _get_labextension_metadata(module)
    base_path = os.path.split(m.__file__)[0]
    full_dests = []
    for labext in labexts:
        src = os.path.join(base_path, labext['src'])
        dest = labext['dest']
        if logger:
            logger.info(f'Installing {src} -> {dest}')
        if not os.path.exists(src):
            build_labextension(base_path, logger=logger)
        full_dest = develop_labextension(src, overwrite=overwrite, symlink=symlink, user=user, sys_prefix=sys_prefix, labextensions_dir=labextensions_dir, destination=dest, logger=logger)
        full_dests.append(full_dest)
    return full_dests