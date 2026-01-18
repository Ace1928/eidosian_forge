import io
import math
import os
import typing
import weakref
def new_page(doc: fitz.Document, pno: int=-1, width: float=595, height: float=842) -> fitz.Page:
    """Create and return a new page object.

    Args:
        pno: (int) insert before this page. Default: after last page.
        width: (float) page width in points. Default: 595 (ISO A4 width).
        height: (float) page height in points. Default 842 (ISO A4 height).
    Returns:
        A fitz.Page object.
    """
    doc._newPage(pno, width=width, height=height)
    return doc[pno]