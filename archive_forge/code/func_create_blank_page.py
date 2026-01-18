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
@staticmethod
def create_blank_page(pdf: Optional[PdfCommonDocProtocol]=None, width: Union[float, Decimal, None]=None, height: Union[float, Decimal, None]=None) -> 'PageObject':
    """
        Return a new blank page.

        If ``width`` or ``height`` is ``None``, try to get the page size
        from the last page of *pdf*.

        Args:
            pdf: PDF file the page belongs to
            width: The width of the new page expressed in default user
                space units.
            height: The height of the new page expressed in default user
                space units.

        Returns:
            The new blank page

        Raises:
            PageSizeNotDefinedError: if ``pdf`` is ``None`` or contains
                no page
        """
    page = PageObject(pdf)
    page.__setitem__(NameObject(PG.TYPE), NameObject('/Page'))
    page.__setitem__(NameObject(PG.PARENT), NullObject())
    page.__setitem__(NameObject(PG.RESOURCES), DictionaryObject())
    if width is None or height is None:
        if pdf is not None and len(pdf.pages) > 0:
            lastpage = pdf.pages[len(pdf.pages) - 1]
            width = lastpage.mediabox.width
            height = lastpage.mediabox.height
        else:
            raise PageSizeNotDefinedError
    page.__setitem__(NameObject(PG.MEDIABOX), RectangleObject((0, 0, width, height)))
    return page