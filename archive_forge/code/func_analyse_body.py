import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def analyse_body(body, env, non_generic):
    for stmt in body:
        if isinstance(stmt, ast.FunctionDef):
            new_type = TypeVariable()
            env[stmt.name] = new_type
    for stmt in body:
        analyse(stmt, env, non_generic)