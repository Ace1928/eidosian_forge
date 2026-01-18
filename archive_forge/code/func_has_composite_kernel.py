import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def has_composite_kernel(self) -> bool:
    return (self.has_composite_implicit_autograd_kernel or self.has_composite_explicit_autograd_kernel or self.has_composite_explicit_autograd_non_functional_kernel) or (self.has_composite_implicit_autograd_kernel and self.has_composite_implicit_autograd_nested_tensor_kernel)