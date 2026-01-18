import io
import math
import os
import typing
import weakref
def get_text_words(page: fitz.Page, clip: rect_like=None, flags: OptInt=None, textpage: fitz.TextPage=None, sort: bool=False, delimiters=None) -> list:
    """Return the text words as a list with the bbox for each word.

    Args:
        flags: (int) control the amount of data parsed into the textpage.
        delimiters: (str,list) characters to use as word delimiters

    Returns:
        Word tuples (x0, y0, x1, y1, "word", bno, lno, wno).
    """
    fitz.CheckParent(page)
    if flags is None:
        flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_MEDIABOX_CLIP
    tp = textpage
    if tp is None:
        tp = page.get_textpage(clip=clip, flags=flags)
    elif getattr(tp, 'parent') != page:
        raise ValueError('not a textpage of this page')
    words = tp.extractWORDS(delimiters)
    if textpage is None:
        del tp
    if sort is True:
        words.sort(key=lambda w: (w[3], w[0]))
    return words