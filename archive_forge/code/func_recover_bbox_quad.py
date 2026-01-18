import io
import math
import os
import typing
import weakref
def recover_bbox_quad(line_dir: tuple, span: dict, bbox: tuple) -> fitz.Quad:
    """Compute the quad located inside the bbox.

    The bbox may be any of the resp. tuples occurring inside the given span.

    Args:
        line_dir: (tuple) 'line["dir"]' of the owning line or None.
        span: (dict) the span. May be from get_texttrace() method.
        bbox: (tuple) the bbox of the span or any of its characters.
    Returns:
        The quad which is wrapped by the bbox.
    """
    if line_dir is None:
        line_dir = span['dir']
    cos, sin = line_dir
    bbox = fitz.Rect(bbox)
    if fitz.TOOLS.set_small_glyph_heights():
        d = 1
    else:
        d = span['ascender'] - span['descender']
    height = d * span['size']
    hs = height * sin
    hc = height * cos
    if hc >= 0 and hs <= 0:
        ul = bbox.bl - (0, hc)
        ur = bbox.tr + (hs, 0)
        ll = bbox.bl - (hs, 0)
        lr = bbox.tr + (0, hc)
    elif hc <= 0 and hs <= 0:
        ul = bbox.br + (hs, 0)
        ur = bbox.tl - (0, hc)
        ll = bbox.br + (0, hc)
        lr = bbox.tl - (hs, 0)
    elif hc <= 0 and hs >= 0:
        ul = bbox.tr - (0, hc)
        ur = bbox.bl + (hs, 0)
        ll = bbox.tr - (hs, 0)
        lr = bbox.bl + (0, hc)
    else:
        ul = bbox.tl + (hs, 0)
        ur = bbox.br - (0, hc)
        ll = bbox.tl + (0, hc)
        lr = bbox.br - (hs, 0)
    return fitz.Quad(ul, ur, ll, lr)