from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def namedtuple_fieldnames(returns: Tuple[Return, ...]) -> List[str]:
    if len(returns) <= 1 or all((r.name is None for r in returns)):
        return []
    else:
        if any((r.name is None for r in returns)):
            raise ValueError('Unnamed field is not supported by codegen')
        return [str(r.name) for r in returns]