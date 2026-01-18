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
def resolve_flag_ref(flag_ref: Union[str, FlagHolder], flag_values: FlagValues) -> Tuple[str, FlagValues]:
    """Helper to validate and resolve a flag reference argument."""
    if isinstance(flag_ref, FlagHolder):
        new_flag_values = flag_ref._flagvalues
        if flag_values != FLAGS and flag_values != new_flag_values:
            raise ValueError('flag_values must not be customized when operating on a FlagHolder')
        return (flag_ref.name, new_flag_values)
    return (flag_ref, flag_values)