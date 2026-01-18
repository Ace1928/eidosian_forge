import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def add_transformation(self, ctm: Union[Transformation, CompressedTransformationMatrix], expand: bool=False) -> None:
    """
        Apply a transformation matrix to the page.

        Args:
            ctm: A 6-element tuple containing the operands of the
                transformation matrix. Alternatively, a
                :py:class:`Transformation<pypdf.Transformation>`
                object can be passed.

        See :doc:`/user/cropping-and-transforming`.
        """
    if isinstance(ctm, Transformation):
        ctm = ctm.ctm
    content = self.get_contents()
    if content is not None:
        content = PageObject._add_transformation_matrix(content, self.pdf, ctm)
        content.isolate_graphics_state()
        self.replace_contents(content)
    if expand:
        corners = [self.mediabox.left.as_numeric(), self.mediabox.bottom.as_numeric(), self.mediabox.left.as_numeric(), self.mediabox.top.as_numeric(), self.mediabox.right.as_numeric(), self.mediabox.top.as_numeric(), self.mediabox.right.as_numeric(), self.mediabox.bottom.as_numeric()]
        ctm = tuple((float(x) for x in ctm))
        new_x = [ctm[0] * corners[i] + ctm[2] * corners[i + 1] + ctm[4] for i in range(0, 8, 2)]
        new_y = [ctm[1] * corners[i] + ctm[3] * corners[i + 1] + ctm[5] for i in range(0, 8, 2)]
        lowerleft = (min(new_x), min(new_y))
        upperright = (max(new_x), max(new_y))
        self.mediabox.lower_left = lowerleft
        self.mediabox.upper_right = upperright