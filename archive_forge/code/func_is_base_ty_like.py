import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def is_base_ty_like(self, base_ty: BaseTy) -> bool:
    return self.elem.is_base_ty_like(base_ty)