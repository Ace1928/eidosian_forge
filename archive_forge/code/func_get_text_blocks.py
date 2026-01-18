import io
import math
import os
import typing
import weakref
def get_text_blocks(page: fitz.Page, clip: rect_like=None, flags: OptInt=None, textpage: fitz.TextPage=None, sort: bool=False) -> list:
    """Return the text blocks on a page.

    Notes:
        Lines in a block are concatenated with line breaks.
    Args:
        flags: (int) control the amount of data parsed into the textpage.
    Returns:
        A list of the blocks. Each item contains the containing rectangle
        coordinates, text lines, block type and running block number.
    """
    fitz.CheckParent(page)
    if flags is None:
        flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_IMAGES | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_MEDIABOX_CLIP
    tp = textpage
    if tp is None:
        tp = page.get_textpage(clip=clip, flags=flags)
    elif getattr(tp, 'parent') != page:
        raise ValueError('not a textpage of this page')
    blocks = tp.extractBLOCKS()
    if textpage is None:
        del tp
    if sort is True:
        blocks.sort(key=lambda b: (b[3], b[0]))
    return blocks