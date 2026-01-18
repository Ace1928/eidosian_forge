import io
import math
import os
import typing
import weakref
def recover_quad(line_dir: tuple, span: dict) -> fitz.Quad:
    """Recover the quadrilateral of a text span.

    Args:
        line_dir: (tuple) 'line["dir"]' of the owning line.
        span: the span.
    Returns:
        The quadrilateral enveloping the span's text.
    """
    if type(line_dir) is not tuple or len(line_dir) != 2:
        raise ValueError('bad line dir argument')
    if type(span) is not dict:
        raise ValueError('bad span argument')
    return recover_bbox_quad(line_dir, span, span['bbox'])