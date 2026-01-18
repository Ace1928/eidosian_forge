import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def remove_inplace(self) -> 'OperatorName':
    return OperatorName(name=BaseOperatorName(base=self.name.base, inplace=False, dunder_method=self.name.dunder_method), overload_name=self.overload_name)