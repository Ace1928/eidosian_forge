from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
def declare_declarations_in_scope(declaration_string, env, private_type=True, *args, **kwargs):
    """
    Declare some declarations given as Cython code in declaration_string
    in scope env.
    """
    CythonUtilityCode(declaration_string, *args, **kwargs).declare_in_scope(env)