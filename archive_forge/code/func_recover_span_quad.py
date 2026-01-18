import io
import math
import os
import typing
import weakref
def recover_span_quad(line_dir: tuple, span: dict, chars: list=None) -> fitz.Quad:
    """Calculate the span quad for 'dict' / 'rawdict' text extractions.

    Notes:
        There are two execution paths:
        1. For the full span quad, the result of 'recover_quad' is returned.
        2. For the quad of a sub-list of characters, the char quads are
           computed and joined. This is only supported for the "rawdict"
           extraction option.

    Args:
        line_dir: (tuple) 'line["dir"]' of the owning line.
        span: (dict) the span.
        chars: (list, optional) sub-list of characters to consider.
    Returns:
        fitz.Quad covering selected characters.
    """
    if line_dir is None:
        line_dir = span['dir']
    if chars is None:
        return recover_quad(line_dir, span)
    if 'chars' not in span.keys():
        raise ValueError("need 'rawdict' option to sub-select chars")
    q0 = recover_char_quad(line_dir, span, chars[0])
    if len(chars) > 1:
        q1 = recover_char_quad(line_dir, span, chars[-1])
    else:
        q1 = q0
    span_ll = q0.ll
    span_lr = q1.lr
    mat0 = fitz.planish_line(span_ll, span_lr)
    x_lr = span_lr * mat0
    small = fitz.TOOLS.set_small_glyph_heights()
    h = span['size'] * (1 if small else span['ascender'] - span['descender'])
    span_rect = fitz.Rect(0, -h, x_lr.x, 0)
    span_quad = span_rect.quad
    span_quad *= ~mat0
    return span_quad