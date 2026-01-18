imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
def get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = handle_special_build(modname, pyxfilename)
    if not extension_mod:
        if not isinstance(pyxfilename, str):
            pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
        from distutils.extension import Extension
        extension_mod = Extension(name=modname, sources=[pyxfilename])
        if language_level is not None:
            extension_mod.cython_directives = {'language_level': language_level}
    return (extension_mod, setup_args)