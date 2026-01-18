from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from mypy.nodes import ARG_NAMED_OPT
from mypy.nodes import Argument
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import MDEF
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneTyp
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType
from . import infer
from . import util
from .names import expr_to_mapped_constructor
from .names import NAMED_TYPE_SQLA_MAPPED
def apply_mypy_mapped_attr(cls: ClassDef, api: SemanticAnalyzerPluginInterface, item: Union[NameExpr, StrExpr], attributes: List[util.SQLAlchemyAttribute]) -> None:
    if isinstance(item, NameExpr):
        name = item.name
    elif isinstance(item, StrExpr):
        name = item.value
    else:
        return None
    for stmt in cls.defs.body:
        if isinstance(stmt, AssignmentStmt) and isinstance(stmt.lvalues[0], NameExpr) and (stmt.lvalues[0].name == name):
            break
    else:
        util.fail(api, f"Can't find mapped attribute {name}", cls)
        return None
    if stmt.type is None:
        util.fail(api, 'Statement linked from _mypy_mapped_attrs has no typing information', stmt)
        return None
    left_hand_explicit_type = get_proper_type(stmt.type)
    assert isinstance(left_hand_explicit_type, (Instance, UnionType, UnboundType))
    attributes.append(util.SQLAlchemyAttribute(name=name, line=item.line, column=item.column, typ=left_hand_explicit_type, info=cls.info))
    apply_type_to_mapped_statement(api, stmt, stmt.lvalues[0], left_hand_explicit_type, None)