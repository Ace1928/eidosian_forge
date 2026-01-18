import math
from functools import lru_cache
from time import monotonic
from typing import Iterable, List, Optional
from .color import Color, blend_rgb
from .color_triplet import ColorTriplet
from .console import Console, ConsoleOptions, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleType
@property
def percentage_completed(self) -> Optional[float]:
    """Calculate percentage complete."""
    if self.total is None:
        return None
    completed = self.completed / self.total * 100.0
    completed = min(100, max(0.0, completed))
    return completed