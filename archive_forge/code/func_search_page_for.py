import io
import math
import os
import typing
import weakref
def search_page_for(doc: fitz.Document, pno: int, text: str, quads: bool=False, clip: rect_like=None, flags: int=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP, textpage: fitz.TextPage=None) -> list:
    """Search for a string on a page.

    Args:
        pno: page number
        text: string to be searched for
        clip: restrict search to this rectangle
        quads: (bool) return quads instead of rectangles
        flags: bit switches, default: join hyphened words
        textpage: reuse a prepared textpage
    Returns:
        a list of rectangles or quads, each containing an occurrence.
    """
    return doc[pno].search_for(text, quads=quads, clip=clip, flags=flags, textpage=textpage)