from __future__ import annotations
from enum import IntEnum
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, TypeVar
import attrs
from attrs import define, field
from fontTools.ufoLib import UFOReader
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.misc import AttrDictMixin
from ufoLib2.serde import serde
from .woff import (
@openTypeGaspRangeRecords.setter
def openTypeGaspRangeRecords(self, value: list[GaspRangeRecord] | None) -> None:
    self._openTypeGaspRangeRecords = _convert_gasp_range_records(value)