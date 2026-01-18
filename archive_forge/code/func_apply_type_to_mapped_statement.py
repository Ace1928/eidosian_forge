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
def apply_type_to_mapped_statement(api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, lvalue: NameExpr, left_hand_explicit_type: Optional[ProperType], python_type_for_type: Optional[ProperType]) -> None:
    """Apply the Mapped[<type>] annotation and right hand object to a
    declarative assignment statement.

    This converts a Python declarative class statement such as::

        class User(Base):
            # ...

            attrname = Column(Integer)

    To one that describes the final Python behavior to Mypy::

        class User(Base):
            # ...

            attrname : Mapped[Optional[int]] = <meaningless temp node>

    """
    left_node = lvalue.node
    assert isinstance(left_node, Var)
    if left_hand_explicit_type is not None:
        lvalue.is_inferred_def = False
        left_node.type = api.named_type(NAMED_TYPE_SQLA_MAPPED, [left_hand_explicit_type])
    else:
        lvalue.is_inferred_def = False
        left_node.type = api.named_type(NAMED_TYPE_SQLA_MAPPED, [AnyType(TypeOfAny.special_form)] if python_type_for_type is None else [python_type_for_type])
    stmt.rvalue = expr_to_mapped_constructor(stmt.rvalue)
    if stmt.type is not None and python_type_for_type is not None:
        stmt.type = python_type_for_type