import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def gets_generated_view_copy(f: NativeFunction) -> bool:
    if not f.is_view_op:
        return False
    if f.has_composite_implicit_autograd_kernel:
        return False
    if 'inplace_view' in f.tags:
        return False
    return True