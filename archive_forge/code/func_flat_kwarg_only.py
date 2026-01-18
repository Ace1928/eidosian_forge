import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def flat_kwarg_only(self) -> Sequence[Argument]:
    ret: List[Argument] = []
    ret.extend(self.pre_tensor_options_kwarg_only)
    if self.tensor_options is not None:
        ret.extend(self.tensor_options.all())
    ret.extend(self.post_tensor_options_kwarg_only)
    return ret