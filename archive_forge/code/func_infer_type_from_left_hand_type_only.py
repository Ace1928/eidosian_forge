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
def infer_type_from_left_hand_type_only(api: SemanticAnalyzerPluginInterface, node: Var, left_hand_explicit_type: Optional[ProperType]) -> Optional[ProperType]:
    """Determine the type based on explicit annotation only.

    if no annotation were present, note that we need one there to know
    the type.

    """
    if left_hand_explicit_type is None:
        msg = "Can't infer type from ORM mapped expression assigned to attribute '{}'; please specify a Python type or Mapped[<python type>] on the left hand side."
        util.fail(api, msg.format(node.name), node)
        return api.named_type(names.NAMED_TYPE_SQLA_MAPPED, [AnyType(TypeOfAny.special_form)])
    else:
        return left_hand_explicit_type