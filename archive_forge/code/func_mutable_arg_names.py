import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def mutable_arg_names(self) -> List[str]:
    return [a.name for a in self.flat_all if a.annotation is not None and a.annotation.is_write]