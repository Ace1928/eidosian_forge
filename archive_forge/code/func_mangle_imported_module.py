from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def mangle_imported_module(module_name):
    return pythran_ward + 'imported__' + module_name.replace('.', '$') + '$'