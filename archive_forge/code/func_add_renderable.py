from collections import defaultdict
from itertools import chain
from operator import itemgetter
from typing import Dict, Iterable, List, Optional, Tuple
from .align import Align, AlignMethod
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .constrain import Constrain
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .table import Table
from .text import TextType
from .jupyter import JupyterMixin
def add_renderable(self, renderable: RenderableType) -> None:
    """Add a renderable to the columns.

        Args:
            renderable (RenderableType): Any renderable object.
        """
    self.renderables.append(renderable)