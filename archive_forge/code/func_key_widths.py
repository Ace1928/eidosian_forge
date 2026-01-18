from __future__ import annotations
import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import remove_missing
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from .guide import GuideElements, guide
@cached_property
def key_widths(self) -> list[float]:
    """
        Widths of the keys

        If legend is vertical, key widths must be equal, so we use the
        maximum. So a plot like

           (ggplot(diamonds, aes(x="cut", y="clarity"))
            + stat_sum(aes(group="cut"))
            + scale_size(range=(3, 25))
           )

        would have keys with variable heights, but fixed width.
        """
    ws = [w for w, _ in self._key_dimensions]
    if self.is_vertical:
        return [max(ws)] * len(ws)
    return ws