import io
import math
import os
import typing
import weakref
def insert_page(doc: fitz.Document, pno: int, text: typing.Union[str, list, None]=None, fontsize: float=11, width: float=595, height: float=842, fontname: str='helv', fontfile: OptStr=None, color: OptSeq=(0,)) -> int:
    """Create a new PDF page and insert some text.

    Notes:
        Function combining fitz.Document.new_page() and fitz.Page.insert_text().
        For parameter details see these methods.
    """
    page = doc.new_page(pno=pno, width=width, height=height)
    if not bool(text):
        return 0
    rc = page.insert_text((50, 72), text, fontsize=fontsize, fontname=fontname, fontfile=fontfile, color=color)
    return rc