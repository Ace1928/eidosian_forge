from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def arg_parser_output_exprs(ps: PythonSignature, f: NativeFunction, *, symint: bool=True) -> Dict[str, PythonArgParserOutputExpr]:
    return {e.name: e for i, a in enumerate(ps.arguments()) for e in (arg_parser_output_expr(i, a, symint=symint),)}