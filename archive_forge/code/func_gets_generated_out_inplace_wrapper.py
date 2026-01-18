import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def gets_generated_out_inplace_wrapper(f: NativeFunction, g: NativeFunctionsGroup, b: BackendIndex) -> bool:
    return f.func.kind() is not SchemaKind.functional and (not b.has_kernel(f)) and b.has_kernel(g.functional)