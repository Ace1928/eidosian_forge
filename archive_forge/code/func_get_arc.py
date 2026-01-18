import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from ..errors import Errors
from ..util import escape_html, minify_html, registry
from .templates import (
def get_arc(self, x_start: int, y: int, y_curve: int, x_end: int) -> str:
    """Render individual arc.

        x_start (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
        x_end (int): X-coordinate of arrow end point.
        RETURNS (str): Definition of the arc path ('d' attribute).
        """
    template = 'M{x},{y} C{x},{c} {e},{c} {e},{y}'
    if self.compact:
        template = 'M{x},{y} {x},{c} {e},{c} {e},{y}'
    return template.format(x=x_start, y=y, c=y_curve, e=x_end)