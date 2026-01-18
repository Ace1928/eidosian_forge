import io
import math
import os
import typing
import weakref
def get_char_widths(doc: fitz.Document, xref: int, limit: int=256, idx: int=0, fontdict: OptDict=None) -> list:
    """Get list of glyph information of a font.

    Notes:
        Must be provided by its XREF number. If we already dealt with the
        font, it will be recorded in doc.FontInfos. Otherwise we insert an
        entry there.
        Finally we return the glyphs for the font. This is a list of
        (glyph, width) where glyph is an integer controlling the char
        appearance, and width is a float controlling the char's spacing:
        width * fontsize is the actual space.
        For 'simple' fonts, glyph == ord(char) will usually be true.
        Exceptions are 'Symbol' and 'ZapfDingbats'. We are providing data for these directly here.
    """
    fontinfo = fitz.CheckFontInfo(doc, xref)
    if fontinfo is None:
        if fontdict is None:
            name, ext, stype, asc, dsc = _get_font_properties(doc, xref)
            fontdict = {'name': name, 'type': stype, 'ext': ext, 'ascender': asc, 'descender': dsc}
        else:
            name = fontdict['name']
            ext = fontdict['ext']
            stype = fontdict['type']
            ordering = fontdict['ordering']
            simple = fontdict['simple']
        if ext == '':
            raise ValueError('xref is not a font')
        if stype in ('Type1', 'MMType1', 'TrueType'):
            simple = True
        else:
            simple = False
        if name in ('Fangti', 'Ming'):
            ordering = 0
        elif name in ('Heiti', 'Song'):
            ordering = 1
        elif name in ('Gothic', 'Mincho'):
            ordering = 2
        elif name in ('Dotum', 'Batang'):
            ordering = 3
        else:
            ordering = -1
        fontdict['simple'] = simple
        if name == 'ZapfDingbats':
            glyphs = fitz.zapf_glyphs
        elif name == 'Symbol':
            glyphs = fitz.symbol_glyphs
        else:
            glyphs = None
        fontdict['glyphs'] = glyphs
        fontdict['ordering'] = ordering
        fontinfo = [xref, fontdict]
        doc.FontInfos.append(fontinfo)
    else:
        fontdict = fontinfo[1]
        glyphs = fontdict['glyphs']
        simple = fontdict['simple']
        ordering = fontdict['ordering']
    if glyphs is None:
        oldlimit = 0
    else:
        oldlimit = len(glyphs)
    mylimit = max(256, limit)
    if mylimit <= oldlimit:
        return glyphs
    if ordering < 0:
        glyphs = doc._get_char_widths(xref, fontdict['name'], fontdict['ext'], fontdict['ordering'], mylimit, idx)
    else:
        glyphs = None
    fontdict['glyphs'] = glyphs
    fontinfo[1] = fontdict
    fitz.UpdateFontInfo(doc, fontinfo)
    return glyphs