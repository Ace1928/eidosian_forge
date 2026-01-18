import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def flat_positional(self) -> Sequence[Argument]:
    ret: List[Argument] = []
    ret.extend(self.pre_self_positional)
    if self.self_arg is not None:
        ret.append(self.self_arg.argument)
    ret.extend(self.post_self_positional)
    return ret