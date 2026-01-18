import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def strip_ret_annotation(r: Return) -> Return:
    return Return(name=r.name if keep_return_names else None, type=r.type, annotation=None)