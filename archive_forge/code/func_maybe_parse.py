import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@staticmethod
def maybe_parse(value: str) -> Optional['ScalarType']:
    for k, v in ScalarType.__members__.items():
        if k == value:
            return v
    return None