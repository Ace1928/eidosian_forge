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
def check_intoin_outtoout(*, l1_arg: Expression, l1_type: Instance, l2_arg: Expression, l2_type: Instance, api: CheckerPluginInterface):
    if l1_type.args[0] != l2_type.args[0]:
        api.fail(f'Layer input ({l1_type.args[0]}) not compatible with next layer input ({l2_type.args[0]})', l1_arg, code=error_layer_input)
        api.fail(f'Layer input ({l2_type.args[0]}) not compatible with previous layer input ({l1_type.args[0]})', l2_arg, code=error_layer_input)
    if l1_type.args[1] != l2_type.args[1]:
        api.fail(f'Layer output ({l1_type.args[1]}) not compatible with next layer output ({l2_type.args[1]})', l1_arg, code=error_layer_output)
        api.fail(f'Layer output ({l2_type.args[1]}) not compatible with previous layer output ({l1_type.args[1]})', l2_arg, code=error_layer_output)