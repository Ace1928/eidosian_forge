import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def insert_textbox(self, rect: rect_like, buffer_: typing.Union[str, list], fontname: OptStr='helv', fontfile: OptStr=None, fontsize: float=11, lineheight: OptFloat=None, set_simple: bool=0, encoding: int=0, color: OptSeq=None, fill: OptSeq=None, expandtabs: int=1, border_width: float=1, align: int=0, render_mode: int=0, rotate: int=0, morph: OptSeq=None, stroke_opacity: float=1, fill_opacity: float=1, oc: int=0) -> float:
    """Insert text into a given rectangle.

        Args:
            rect -- the textbox to fill
            buffer_ -- text to be inserted
            fontname -- a Base-14 font, font name or '/name'
            fontfile -- name of a font file
            fontsize -- font size
            lineheight -- overwrite the font property
            color -- RGB stroke color triple
            fill -- RGB fill color triple
            render_mode -- text rendering control
            border_width -- thickness of glyph borders
            expandtabs -- handles tabulators with string function
            align -- left, center, right, justified
            rotate -- 0, 90, 180, or 270 degrees
            morph -- morph box with a matrix and a fixpoint
        Returns:
            unused or deficit rectangle area (float)
        """
    rect = Rect(rect)
    if rect.is_empty or rect.is_infinite:
        raise ValueError('text box must be finite and not empty')
    color_str = ColorCode(color, 'c')
    fill_str = ColorCode(fill, 'f')
    if fill is None and render_mode == 0:
        fill = color
        fill_str = ColorCode(color, 'f')
    optcont = self.page._get_optional_content(oc)
    if optcont is not None:
        bdc = '/OC /%s BDC\n' % optcont
        emc = 'EMC\n'
    else:
        bdc = emc = ''
    alpha = self.page._set_opacity(CA=stroke_opacity, ca=fill_opacity)
    if alpha is None:
        alpha = ''
    else:
        alpha = '/%s gs\n' % alpha
    if rotate % 90 != 0:
        raise ValueError('rotate must be multiple of 90')
    rot = rotate
    while rot < 0:
        rot += 360
    rot = rot % 360
    if not bool(buffer_):
        return rect.height if rot in (0, 180) else rect.width
    cmp90 = '0 1 -1 0 0 0 cm\n'
    cmm90 = '0 -1 1 0 0 0 cm\n'
    cm180 = '-1 0 0 -1 0 0 cm\n'
    height = self.height
    fname = fontname
    if fname.startswith('/'):
        fname = fname[1:]
    xref = self.page.insert_font(fontname=fname, fontfile=fontfile, encoding=encoding, set_simple=set_simple)
    fontinfo = CheckFontInfo(self.doc, xref)
    fontdict = fontinfo[1]
    ordering = fontdict['ordering']
    simple = fontdict['simple']
    glyphs = fontdict['glyphs']
    bfname = fontdict['name']
    ascender = fontdict['ascender']
    descender = fontdict['descender']
    if lineheight:
        lheight_factor = lineheight
    elif ascender - descender <= 1:
        lheight_factor = 1.2
    else:
        lheight_factor = ascender - descender
    lheight = fontsize * lheight_factor
    if type(buffer_) in (list, tuple):
        t0 = '\n'.join(buffer_)
    else:
        t0 = buffer_
    maxcode = max([ord(c) for c in t0])
    if simple and maxcode > 255:
        t0 = ''.join([c if ord(c) < 256 else '?' for c in t0])
    t0 = t0.splitlines()
    glyphs = self.doc.get_char_widths(xref, maxcode + 1)
    if simple and bfname not in ('Symbol', 'ZapfDingbats'):
        tj_glyphs = None
    else:
        tj_glyphs = glyphs

    def pixlen(x):
        """Calculate pixel length of x."""
        if ordering < 0:
            return sum([glyphs[ord(c)][1] for c in x]) * fontsize
        else:
            return len(x) * fontsize
    if ordering < 0:
        blen = glyphs[32][1] * fontsize
    else:
        blen = fontsize
    text = ''
    if CheckMorph(morph):
        m1 = Matrix(1, 0, 0, 1, morph[0].x + self.x, self.height - morph[0].y - self.y)
        mat = ~m1 * morph[1] * m1
        cm = '%g %g %g %g %g %g cm\n' % JM_TUPLE(mat)
    else:
        cm = ''
    progr = 1
    c_pnt = Point(0, fontsize * ascender)
    if rot == 0:
        point = rect.tl + c_pnt
        pos = point.y + self.y
        maxwidth = rect.width
        maxpos = rect.y1 + self.y
    elif rot == 90:
        c_pnt = Point(fontsize * ascender, 0)
        point = rect.bl + c_pnt
        pos = point.x + self.x
        maxwidth = rect.height
        maxpos = rect.x1 + self.x
        cm += cmp90
    elif rot == 180:
        c_pnt = -Point(0, fontsize * ascender)
        point = rect.br + c_pnt
        pos = point.y + self.y
        maxwidth = rect.width
        progr = -1
        maxpos = rect.y0 + self.y
        cm += cm180
    else:
        c_pnt = -Point(fontsize * ascender, 0)
        point = rect.tr + c_pnt
        pos = point.x + self.x
        maxwidth = rect.height
        progr = -1
        maxpos = rect.x0 + self.x
        cm += cmm90
    just_tab = []
    for i, line in enumerate(t0):
        line_t = line.expandtabs(expandtabs).split(' ')
        lbuff = ''
        rest = maxwidth
        for word in line_t:
            pl_w = pixlen(word)
            if rest >= pl_w:
                lbuff += word + ' '
                rest -= pl_w + blen
                continue
            if len(lbuff) > 0:
                lbuff = lbuff.rstrip() + '\n'
                text += lbuff
                pos += lheight * progr
                just_tab.append(True)
                lbuff = ''
            rest = maxwidth
            if pl_w <= maxwidth:
                lbuff = word + ' '
                rest = maxwidth - pl_w - blen
                continue
            if len(just_tab) > 0:
                just_tab[-1] = False
            for c in word:
                if pixlen(lbuff) <= maxwidth - pixlen(c):
                    lbuff += c
                else:
                    lbuff += '\n'
                    text += lbuff
                    pos += lheight * progr
                    just_tab.append(False)
                    lbuff = c
            lbuff += ' '
            rest = maxwidth - pixlen(lbuff)
        if lbuff != '':
            text += lbuff.rstrip()
            just_tab.append(False)
        if i < len(t0) - 1:
            text += '\n'
            pos += lheight * progr
    more = (pos - maxpos) * progr
    if more > EPSILON:
        return -1 * more
    more = abs(more)
    if more < EPSILON:
        more = 0
    nres = '\nq\n%s%sBT\n' % (bdc, alpha) + cm
    templ = '1 0 0 1 %g %g Tm /%s %g Tf '
    text_t = text.splitlines()
    just_tab[-1] = False
    for i, t in enumerate(text_t):
        pl = maxwidth - pixlen(t)
        pnt = point + c_pnt * (i * lheight_factor)
        if align == 1:
            if rot in (0, 180):
                pnt = pnt + Point(pl / 2, 0) * progr
            else:
                pnt = pnt - Point(0, pl / 2) * progr
        elif align == 2:
            if rot in (0, 180):
                pnt = pnt + Point(pl, 0) * progr
            else:
                pnt = pnt - Point(0, pl) * progr
        elif align == 3:
            spaces = t.count(' ')
            if spaces > 0 and just_tab[i]:
                spacing = pl / spaces
            else:
                spacing = 0
        top = height - pnt.y - self.y
        left = pnt.x + self.x
        if rot == 90:
            left = height - pnt.y - self.y
            top = -pnt.x - self.x
        elif rot == 270:
            left = -height + pnt.y + self.y
            top = pnt.x + self.x
        elif rot == 180:
            left = -pnt.x - self.x
            top = -height + pnt.y + self.y
        nres += templ % (left, top, fname, fontsize)
        if render_mode > 0:
            nres += '%i Tr ' % render_mode
        if align == 3:
            nres += '%g Tw ' % spacing
        if color is not None:
            nres += color_str
        if fill is not None:
            nres += fill_str
        if border_width != 1:
            nres += '%g w ' % border_width
        nres += '%sTJ\n' % getTJstr(t, tj_glyphs, simple, ordering)
    nres += 'ET\n%sQ\n' % emc
    self.text_cont += nres
    self.updateRect(rect)
    return more