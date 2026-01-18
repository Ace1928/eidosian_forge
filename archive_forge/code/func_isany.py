import heapq
import logging
from typing import (
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdfinterp import Color
from .pdfinterp import PDFGraphicState
from .pdftypes import PDFStream
from .utils import INF, PathSegment
from .utils import LTComponentT
from .utils import Matrix
from .utils import Plane
from .utils import Point
from .utils import Rect
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import fsplit
from .utils import get_bound
from .utils import matrix2str
from .utils import uniq
def isany(obj1: ElementT, obj2: ElementT) -> Set[ElementT]:
    """Check if there's any other object between obj1 and obj2."""
    x0 = min(obj1.x0, obj2.x0)
    y0 = min(obj1.y0, obj2.y0)
    x1 = max(obj1.x1, obj2.x1)
    y1 = max(obj1.y1, obj2.y1)
    objs = set(plane.find((x0, y0, x1, y1)))
    return objs.difference((obj1, obj2))