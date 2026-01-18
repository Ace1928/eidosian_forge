from __future__ import annotations
import logging # isort:skip
import math
from collections import defaultdict
from typing import (
from .core.enums import Location, LocationType, SizingModeType
from .core.property.singletons import Undefined, UndefinedType
from .models import (
from .util.dataclasses import dataclass
from .util.warnings import warn
def assert_unique(values: list[V], name: ToolbarOptions) -> V | UndefinedType:
    if name in toolbar_options:
        return toolbar_options[name]
    n = len(set(values))
    if n == 0:
        return Undefined
    elif n > 1:
        warn(f"found multiple competing values for 'toolbar.{name}' property; using the latest value")
    return values[-1]