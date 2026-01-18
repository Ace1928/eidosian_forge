import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def remove_self_annotation(self) -> 'Arguments':
    assert self.self_arg is not None
    return dataclasses.replace(self, self_arg=SelfArgument(dataclasses.replace(self.self_arg.argument, annotation=None)))