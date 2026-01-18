from __future__ import annotations
from typing import Optional
from typing import Sequence
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import LambdaExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.subtypes import is_subtype
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnionType
from . import names
from . import util
def extract_python_type_from_typeengine(api: SemanticAnalyzerPluginInterface, node: TypeInfo, type_args: Sequence[Expression]) -> ProperType:
    if node.fullname == 'sqlalchemy.sql.sqltypes.Enum' and type_args:
        first_arg = type_args[0]
        if isinstance(first_arg, RefExpr) and isinstance(first_arg.node, TypeInfo):
            for base_ in first_arg.node.mro:
                if base_.fullname == 'enum.Enum':
                    return Instance(first_arg.node, [])
        else:
            return api.named_type(names.NAMED_TYPE_BUILTINS_STR, [])
    assert node.has_base('sqlalchemy.sql.type_api.TypeEngine'), 'could not extract Python type from node: %s' % node
    type_engine_sym = api.lookup_fully_qualified_or_none('sqlalchemy.sql.type_api.TypeEngine')
    assert type_engine_sym is not None and isinstance(type_engine_sym.node, TypeInfo)
    type_engine = map_instance_to_supertype(Instance(node, []), type_engine_sym.node)
    return get_proper_type(type_engine.args[-1])