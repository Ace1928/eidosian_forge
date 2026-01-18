from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import Decorator
from mypy.nodes import LambdaExpr
from mypy.nodes import ListExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import PlaceholderNode
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolNode
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import Type
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType
from . import apply
from . import infer
from . import names
from . import util
def scan_declarative_assignments_and_apply_types(cls: ClassDef, api: SemanticAnalyzerPluginInterface, is_mixin_scan: bool=False) -> Optional[List[util.SQLAlchemyAttribute]]:
    info = util.info_for_cls(cls, api)
    if info is None:
        return None
    elif cls.fullname.startswith('builtins'):
        return None
    mapped_attributes: Optional[List[util.SQLAlchemyAttribute]] = util.get_mapped_attributes(info, api)
    util.establish_as_sqlalchemy(info)
    if mapped_attributes is not None:
        if not is_mixin_scan:
            apply.re_apply_declarative_assignments(cls, api, mapped_attributes)
        return mapped_attributes
    mapped_attributes = []
    if not cls.defs.body:
        for sym_name, sym in info.names.items():
            _scan_symbol_table_entry(cls, api, sym_name, sym, mapped_attributes)
    else:
        for stmt in util.flatten_typechecking(cls.defs.body):
            if isinstance(stmt, AssignmentStmt):
                _scan_declarative_assignment_stmt(cls, api, stmt, mapped_attributes)
            elif isinstance(stmt, Decorator):
                _scan_declarative_decorator_stmt(cls, api, stmt, mapped_attributes)
    _scan_for_mapped_bases(cls, api)
    if not is_mixin_scan:
        apply.add_additional_orm_attributes(cls, api, mapped_attributes)
    util.set_mapped_attributes(info, mapped_attributes)
    return mapped_attributes