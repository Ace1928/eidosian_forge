from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def inline_defn_ctor(self) -> str:
    args_str = ', '.join((a.decl() for a in self.arguments().ctor))
    init_str = ', '.join((f'{a.name}_({a.name})' for a in self.arguments().ctor))
    return f'{self.name}({args_str}) : {init_str} {{}}'