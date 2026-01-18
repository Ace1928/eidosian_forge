from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ...util.dataclasses import Unspecified
from ...util.serialization import convert_datetime_type, convert_timedelta_type
from ...util.strings import nice_join
from .. import enums
from .color import ALPHA_DEFAULT_HELP, COLOR_DEFAULT_HELP, Color
from .datetime import Datetime, TimeDelta
from .descriptors import DataSpecPropertyDescriptor, UnitsSpecPropertyDescriptor
from .either import Either
from .enum import Enum
from .instance import Instance
from .nothing import Nothing
from .nullable import Nullable
from .primitive import (
from .serialized import NotSerialized
from .singletons import Undefined
from .struct import Optional, Struct
from .vectorization import (
from .visual import (
@classmethod
def is_color_tuple_shape(cls, val):
    """ Whether the value is the correct shape to be a color tuple

        Checks for a 3 or 4-tuple of numbers

        Args:
            val (str) : the value to check

        Returns:
            True, if the value could be a color tuple

        """
    return isinstance(val, tuple) and len(val) in (3, 4) and all((isinstance(v, (float, int)) for v in val))