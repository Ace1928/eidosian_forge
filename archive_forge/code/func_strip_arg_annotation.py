import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def strip_arg_annotation(a: Argument) -> Argument:
    return Argument(name=a.name, type=a.type, default=a.default if not strip_default else None, annotation=None)