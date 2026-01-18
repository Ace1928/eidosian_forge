from __future__ import annotations
import re
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type as TypingType
from typing import TypeVar
from typing import Union
from mypy import version
from mypy.messages import format_type as _mypy_format_type
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import CLASSDEF_NO_INFO
from mypy.nodes import Context
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import IfStmt
from mypy.nodes import JsonDict
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import Statement
from mypy.nodes import SymbolTableNode
from mypy.nodes import TypeAlias
from mypy.nodes import TypeInfo
from mypy.options import Options
from mypy.plugin import ClassDefContext
from mypy.plugin import DynamicClassDefContext
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.plugins.common import deserialize_and_fixup_type
from mypy.typeops import map_type_from_supertype
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import Type
from mypy.types import TypeVarType
from mypy.types import UnboundType
from mypy.types import UnionType
def has_declarative_base(info: TypeInfo) -> bool:
    is_base = _get_info_mro_metadata(info, 'is_base')
    return is_base is True