from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def signature_str_pyi(self, *, skip_outputs: bool=False) -> str:
    args = self.arguments(skip_outputs=skip_outputs)
    schema_formals: List[str] = [a.argument_str_pyi(method=self.method, deprecated=True) for a in args]
    positional_argc = len(self.input_args)
    if len(schema_formals) > positional_argc:
        schema_formals.insert(positional_argc, '*')
    returns_str = returns_str_pyi(self)
    return f'def {self.name}({', '.join(schema_formals)}) -> {returns_str}: ...'