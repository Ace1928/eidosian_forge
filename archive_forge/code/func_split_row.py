from abc import ABC, abstractmethod
from itertools import islice
from operator import itemgetter
from threading import RLock
from typing import (
from ._ratio import ratio_resolve
from .align import Align
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .highlighter import ReprHighlighter
from .panel import Panel
from .pretty import Pretty
from .region import Region
from .repr import Result, rich_repr
from .segment import Segment
from .style import StyleType
def split_row(self, *layouts: Union['Layout', RenderableType]) -> None:
    """Split the layout in to a row (layouts side by side).

        Args:
            *layouts (Layout): Positional arguments should be (sub) Layout instances.
        """
    self.split(*layouts, splitter='row')