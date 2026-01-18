import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def extend_path(path, name):
    """Helper so breezy.plugins can be a sort of namespace package.

    To be used in similar fashion to pkgutil.extend_path:

        from breezy.plugins import extend_path
        __path__ = extend_path(__path__, __name__)

    Inspects the BRZ_PLUGIN* envvars, sys.path, and the filesystem to find
    plugins. May mutate sys.modules in order to block plugin loading, and may
    append a new meta path finder to sys.meta_path for plugins@ loading.

    Returns a list of paths to import from, as an enhanced object that also
    contains details of the other configuration used.
    """
    blocks = _env_disable_plugins()
    _block_plugins(blocks)
    extra_details = _env_plugins_at()
    _install_importer_if_needed(extra_details)
    paths = _iter_plugin_paths(_env_plugin_path(), path)
    return _Path(name, blocks, extra_details, paths)