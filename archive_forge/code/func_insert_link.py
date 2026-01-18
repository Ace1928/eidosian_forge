import io
import math
import os
import typing
import weakref
def insert_link(page: fitz.Page, lnk: dict, mark: bool=True) -> None:
    """Insert a new link for the current page."""
    fitz.CheckParent(page)
    annot = getLinkText(page, lnk)
    if annot == '':
        raise ValueError('link kind not supported')
    page._addAnnot_FromString((annot,))