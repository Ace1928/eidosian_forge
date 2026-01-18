from pythran.analyses import Globals, Ancestors
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import attributes, functions, methods, MODULES
from pythran.tables import duplicated_methods
from pythran.conversion import mangle, demangle
from pythran.utils import isstr
import gast as ast
from functools import reduce
@staticmethod
def renamer(v, cur_module):
    """
        Rename function path to fit Pythonic naming.
        """
    mname = demangle(v)
    return (v, mname)