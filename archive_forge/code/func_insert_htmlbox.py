import io
import math
import os
import typing
import weakref
def insert_htmlbox(page, rect, text, *, css=None, scale_low=0, archive=None, rotate=0, oc=0, opacity=1, overlay=True) -> float:
    """Insert text with optional HTML tags and stylings into a rectangle.

    Args:
        rect: (rect-like) rectangle into which the text should be placed.
        text: (str) text with optional HTML tags and stylings.
        css: (str) CSS styling commands.
        scale_low: (float) force-fit content by scaling it down. Must be in
            range [0, 1]. If 1, no scaling will take place. If 0, arbitrary
            down-scaling is acceptable. A value of 0.1 would mean that content
            may be scaled down by at most 90%.
        archive: Archive object pointing to locations of used fonts or images
        rotate: (int) rotate the text in the box by a multiple of 90 degrees.
        oc: (int) the xref of an OCG / OCMD (Optional Content).
        opacity: (float) set opacity of inserted content.
        overlay: (bool) put text on top of page content.
    Returns:
        A tuple of floats (spare_height, scale).
        spare_height: -1 if content did not fit, else >= 0. It is the height of the
               unused (still available) rectangle stripe. Positive only if
               scale_min = 1 (no down scaling).
        scale: downscaling factor, 0 < scale <= 1. Set to 0 if spare_height = -1 (no fit).
    """
    if not rotate % 90 == 0:
        raise ValueError('bad rotation angle')
    while rotate < 0:
        rotate += 360
    while rotate >= 360:
        rotate -= 360
    if not 0 <= scale_low <= 1:
        raise ValueError("'scale_low' must be in [0, 1]")
    if css is None:
        css = ''
    rect = fitz.Rect(rect)
    if rotate in (90, 270):
        temp_rect = fitz.Rect(0, 0, rect.height, rect.width)
    else:
        temp_rect = fitz.Rect(0, 0, rect.width, rect.height)
    mycss = 'body {margin:1px;}' + css
    if isinstance(text, str):
        story = fitz.Story(html=text, user_css=mycss, archive=archive)
    elif isinstance(text, fitz.Story):
        story = text
    else:
        raise ValueError("'text' must be a string or a Story")
    scale_max = None if scale_low == 0 else 1 / scale_low
    fit = story.fit_scale(temp_rect, scale_min=1, scale_max=scale_max)
    if fit.big_enough is False:
        return (-1, scale_low)
    filled = fit.filled
    scale = 1 / fit.parameter
    spare_height = fit.rect.y1 - filled[3]
    if scale != 1 or spare_height < 0:
        spare_height = 0

    def rect_function(*args):
        return (fit.rect, fit.rect, fitz.Identity)
    doc = story.write_with_links(rect_function)
    if 0 <= opacity < 1:
        tpage = doc[0]
        alp0 = tpage._set_opacity(CA=opacity, ca=opacity)
        s = f'/{alp0} gs\n'
        fitz.TOOLS._insert_contents(tpage, s.encode(), 0)
    page.show_pdf_page(rect, doc, 0, rotate=rotate, oc=oc, overlay=overlay)
    mp1 = (fit.rect.tl + fit.rect.br) / 2 * scale
    mp2 = (rect.tl + rect.br) / 2
    mat = fitz.Matrix(scale, 0, 0, scale, -mp1.x, -mp1.y) * fitz.Matrix(-rotate) * fitz.Matrix(1, 0, 0, 1, mp2.x, mp2.y)
    for link in doc[0].get_links():
        link['from'] *= mat
        page.insert_link(link)
    return (spare_height, scale)