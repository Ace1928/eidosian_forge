import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def listmodules(self, key=''):
    if key:
        self.output.write("\nHere is a list of modules whose name or summary contains '{}'.\nIf there are any, enter a module name to get more help.\n\n".format(key))
        apropos(key)
    else:
        self.output.write('\nPlease wait a moment while I gather a list of all available modules...\n\n')
        modules = {}

        def callback(path, modname, desc, modules=modules):
            if modname and modname[-9:] == '.__init__':
                modname = modname[:-9] + ' (package)'
            if modname.find('.') < 0:
                modules[modname] = 1

        def onerror(modname):
            callback(None, modname, None)
        ModuleScanner().run(callback, onerror=onerror)
        self.list(modules.keys())
        self.output.write('\nEnter any module name to get more help.  Or, type "modules spam" to search\nfor modules whose name or summary contain the string "spam".\n')