import copy
import itertools
import logging
import os
import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Text, TextIO, Generic, TypeVar, Union, Tuple
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
from absl.flags._flag import Flag
def resolve_flag_refs(flag_refs: Sequence[Union[str, FlagHolder]], flag_values: FlagValues) -> Tuple[List[str], FlagValues]:
    """Helper to validate and resolve flag reference list arguments."""
    fv = None
    names = []
    for ref in flag_refs:
        if isinstance(ref, FlagHolder):
            newfv = ref._flagvalues
            name = ref.name
        else:
            newfv = flag_values
            name = ref
        if fv and fv != newfv:
            raise ValueError('multiple FlagValues instances used in invocation. FlagHolders must be registered to the same FlagValues instance as do flag names, if provided.')
        fv = newfv
        names.append(name)
    return (names, fv)