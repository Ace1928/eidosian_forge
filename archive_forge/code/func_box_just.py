from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import ensure_xy_location
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import (
from ..mapping.aes import rename_aesthetics
from .guide import guide
@cached_property
def box_just(self) -> TextJustification:
    if not (box_just := self.theme.getp('legend_box_just')):
        box_just = 'left' if self.position in {'left', 'right'} else 'right'
    return box_just