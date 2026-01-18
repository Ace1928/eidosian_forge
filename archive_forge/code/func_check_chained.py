import itertools
from typing import Dict, List
from mypy.checker import TypeChecker
from mypy.errorcodes import ErrorCode
from mypy.errors import Errors
from mypy.nodes import CallExpr, Decorator, Expression, FuncDef, MypyFile, NameExpr
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, FunctionContext, Plugin
from mypy.subtypes import is_subtype
from mypy.types import CallableType, Instance, Type, TypeVarType
def check_chained(*, l1_arg: Expression, l1_type: Instance, l2_arg: Expression, l2_type: Instance, api: CheckerPluginInterface):
    if not is_subtype(l1_type.args[1], l2_type.args[0]):
        api.fail(f'Layer outputs type ({l1_type.args[1]}) but the next layer expects ({l2_type.args[0]}) as an input', l1_arg, code=error_layer_output)
        api.fail(f'Layer input type ({l2_type.args[0]}) is not compatible with output ({l1_type.args[1]}) from previous layer', l2_arg, code=error_layer_input)