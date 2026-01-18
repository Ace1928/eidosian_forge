from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def returns_named_tuple_pyi(signature: PythonSignature) -> Optional[Tuple[str, str]]:
    python_returns = [return_type_str_pyi(r.type) for r in signature.returns.returns]
    namedtuple_name = signature.name
    field_names = namedtuple_fieldnames(signature.returns.returns)
    if field_names:
        namedtuple_def_lines = [f'class {namedtuple_name}(NamedTuple):']
        namedtuple_def_lines.extend((f'    {name}: {typ}' for name, typ in zip(field_names, python_returns)))
        namedtuple_def_lines.append('')
        namedtuple_def = '\n'.join(namedtuple_def_lines)
        return (namedtuple_name, namedtuple_def)
    return None