import io
import math
import os
import typing
import weakref
def center_rect(annot_rect, new_text, font, fsize):
    """Calculate minimal sub-rectangle for the overlay text.

        Notes:
            Because 'insert_textbox' supports no vertical text centering,
            we calculate an approximate number of lines here and return a
            sub-rect with smaller height, which should still be sufficient.
        Args:
            annot_rect: the annotation rectangle
            new_text: the text to insert.
            font: the fontname. Must be one of the CJK or Base-14 set, else
                the rectangle is returned unchanged.
            fsize: the fontsize
        Returns:
            A rectangle to use instead of the annot rectangle.
        """
    exception_types = (ValueError, mupdf.FzErrorBase)
    if fitz.mupdf_version_tuple < (1, 24):
        exception_types = ValueError
    if not new_text:
        return annot_rect
    try:
        text_width = fitz.get_text_length(new_text, font, fsize)
    except exception_types:
        if g_exceptions_verbose:
            fitz.exception_info()
        return annot_rect
    line_height = fsize * 1.2
    limit = annot_rect.width
    h = math.ceil(text_width / limit) * line_height
    if h >= annot_rect.height:
        return annot_rect
    r = annot_rect
    y = (annot_rect.tl.y + annot_rect.bl.y - h) * 0.5
    r.y0 = y
    return r