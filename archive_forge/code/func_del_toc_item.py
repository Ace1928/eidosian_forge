import io
import math
import os
import typing
import weakref
def del_toc_item(doc: fitz.Document, idx: int) -> None:
    """Delete TOC / bookmark item by index."""
    xref = doc.get_outline_xrefs()[idx]
    doc._remove_toc_item(xref)