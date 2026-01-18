import io
import math
import os
import typing
import weakref
def get_page_pixmap(doc: fitz.Document, pno: int, *, matrix: matrix_like=fitz.Identity, dpi=None, colorspace: fitz.Colorspace=fitz.csRGB, clip: rect_like=None, alpha: bool=False, annots: bool=True) -> fitz.Pixmap:
    """Create pixmap of document page by page number.

    Notes:
        Convenience function calling page.get_pixmap.
    Args:
        pno: (int) page number
        matrix: fitz.Matrix for transformation (default: fitz.Identity).
        colorspace: (str,fitz.Colorspace) rgb, rgb, gray - case ignored, default csRGB.
        clip: (irect-like) restrict rendering to this area.
        alpha: (bool) include alpha channel
        annots: (bool) also render annotations
    """
    return doc[pno].get_pixmap(matrix=matrix, dpi=dpi, colorspace=colorspace, clip=clip, alpha=alpha, annots=annots)