from typing import Dict, List, Optional, Tuple, Union
from .._utils import StreamType, deprecate_with_replacement
from ..constants import OutlineFontFlag
from ._base import (
from ._data_structures import (
from ._fit import Fit
from ._outline import OutlineItem
from ._rectangle import RectangleObject
from ._utils import (
from ._viewerpref import ViewerPreferences
@staticmethod
def polyline(vertices: List[Tuple[float, float]]) -> DictionaryObject:
    """
        Draw a polyline on the PDF.

        Args:
            vertices: Array specifying the vertices (x, y) coordinates of the poly-line.

        Returns:
            A dictionary object representing the annotation.
        """
    deprecate_with_replacement('AnnotationBuilder.polyline', 'pypdf.annotations.PolyLine', '4.0.0')
    from ..annotations import PolyLine
    return PolyLine(vertices=vertices)