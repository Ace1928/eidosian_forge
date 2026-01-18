from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
def iterAttributes(self):
    """
        List all the attributes defined in this module.

        Note: Future work is planned here to make it possible to list python
        attributes on a module without loading the module by inspecting ASTs or
        bytecode, but currently any iteration of PythonModule objects insists
        they must be loaded, and will use inspect.getmodule.

        @raise NotImplementedError: if this module is not loaded.

        @return: a generator yielding PythonAttribute instances describing the
        attributes of this module.
        """
    if not self.isLoaded():
        raise NotImplementedError("You can't load attributes from non-loaded modules yet.")
    for name, val in inspect.getmembers(self.load()):
        yield PythonAttribute(self.name + '.' + name, self, True, val)