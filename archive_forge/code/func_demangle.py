from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def demangle(name):
    return name[len(pythran_ward + 'imported__'):-1].replace('$', '.')