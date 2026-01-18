import logging
import re
from io import BytesIO
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .pdfcolor import PDFColorSpace
from .pdfcolor import PREDEFINED_COLORSPACE
from .pdfdevice import PDFDevice
from .pdfdevice import PDFTextSeq
from .pdffont import PDFCIDFont
from .pdffont import PDFFont
from .pdffont import PDFFontError
from .pdffont import PDFTrueTypeFont
from .pdffont import PDFType1Font
from .pdffont import PDFType3Font
from .pdfpage import PDFPage
from .pdftypes import PDFException
from .pdftypes import PDFObjRef
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral, PSTypeError
from .psparser import PSStackParser
from .psparser import PSStackType
from .psparser import keyword_name
from .psparser import literal_name
from .utils import MATRIX_IDENTITY
from .utils import Matrix, Point, PathSegment, Rect
from .utils import choplist
from .utils import mult_matrix
def do_TD(self, tx: PDFStackT, ty: PDFStackT) -> None:
    """Move text position and set leading"""
    tx = cast(float, tx)
    ty = cast(float, ty)
    a, b, c, d, e, f = self.textstate.matrix
    self.textstate.matrix = (a, b, c, d, tx * a + ty * c + e, tx * b + ty * d + f)
    self.textstate.leading = ty
    self.textstate.linematrix = (0, 0)
    return