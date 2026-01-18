imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
def handle_special_build(modname, pyxfilename):
    special_build = os.path.splitext(pyxfilename)[0] + PYXBLD_EXT
    ext = None
    setup_args = {}
    if os.path.exists(special_build):
        mod = load_source(special_build)
        make_ext = getattr(mod, 'make_ext', None)
        if make_ext:
            ext = make_ext(modname, pyxfilename)
            assert ext and ext.sources, 'make_ext in %s did not return Extension' % special_build
        make_setup_args = getattr(mod, 'make_setup_args', None)
        if make_setup_args:
            setup_args = make_setup_args()
            assert isinstance(setup_args, dict), 'make_setup_args in %s did not return a dict' % special_build
        assert ext or setup_args, 'neither make_ext nor make_setup_args %s' % special_build
        ext.sources = [os.path.join(os.path.dirname(special_build), source) for source in ext.sources]
    return (ext, setup_args)