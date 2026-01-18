import io
import math
import os
import typing
import weakref
def search_for(page, text, *, clip=None, quads=False, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_MEDIABOX_CLIP, textpage=None) -> list:
    """Search for a string on a page.

    Args:
        text: string to be searched for
        clip: restrict search to this rectangle
        quads: (bool) return quads instead of rectangles
        flags: bit switches, default: join hyphened words
        textpage: a pre-created fitz.TextPage
    Returns:
        a list of rectangles or quads, each containing one occurrence.
    """
    if clip is not None:
        clip = fitz.Rect(clip)
    fitz.CheckParent(page)
    tp = textpage
    if tp is None:
        tp = page.get_textpage(clip=clip, flags=flags)
    elif getattr(tp, 'parent') != page:
        raise ValueError('not a textpage of this page')
    rlist = tp.search(text, quads=quads)
    if textpage is None:
        del tp
    return rlist