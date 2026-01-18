import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def embedTTF(self, filename, characters):
    """Embed the TTF font from the named file into the document."""
    font = get_font(filename)
    fonttype = mpl.rcParams['pdf.fonttype']

    def cvt(length, upe=font.units_per_EM, nearest=True):
        """Convert font coordinates to PDF glyph coordinates."""
        value = length / upe * 1000
        if nearest:
            return round(value)
        if value < 0:
            return math.floor(value)
        else:
            return math.ceil(value)

    def embedTTFType3(font, characters, descriptor):
        """The Type 3-specific part of embedding a Truetype font"""
        widthsObject = self.reserveObject('font widths')
        fontdescObject = self.reserveObject('font descriptor')
        fontdictObject = self.reserveObject('font dictionary')
        charprocsObject = self.reserveObject('character procs')
        differencesArray = []
        firstchar, lastchar = (0, 255)
        bbox = [cvt(x, nearest=False) for x in font.bbox]
        fontdict = {'Type': Name('Font'), 'BaseFont': ps_name, 'FirstChar': firstchar, 'LastChar': lastchar, 'FontDescriptor': fontdescObject, 'Subtype': Name('Type3'), 'Name': descriptor['FontName'], 'FontBBox': bbox, 'FontMatrix': [0.001, 0, 0, 0.001, 0, 0], 'CharProcs': charprocsObject, 'Encoding': {'Type': Name('Encoding'), 'Differences': differencesArray}, 'Widths': widthsObject}
        from encodings import cp1252

        def get_char_width(charcode):
            s = ord(cp1252.decoding_table[charcode])
            width = font.load_char(s, flags=LOAD_NO_SCALE | LOAD_NO_HINTING).horiAdvance
            return cvt(width)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            widths = [get_char_width(charcode) for charcode in range(firstchar, lastchar + 1)]
        descriptor['MaxWidth'] = max(widths)
        glyph_ids = []
        differences = []
        multi_byte_chars = set()
        for c in characters:
            ccode = c
            gind = font.get_char_index(ccode)
            glyph_ids.append(gind)
            glyph_name = font.get_glyph_name(gind)
            if ccode <= 255:
                differences.append((ccode, glyph_name))
            else:
                multi_byte_chars.add(glyph_name)
        differences.sort()
        last_c = -2
        for c, name in differences:
            if c != last_c + 1:
                differencesArray.append(c)
            differencesArray.append(Name(name))
            last_c = c
        rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
        charprocs = {}
        for charname in sorted(rawcharprocs):
            stream = rawcharprocs[charname]
            charprocDict = {}
            if charname in multi_byte_chars:
                charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
                stream = stream[stream.find(b'd1') + 2:]
            charprocObject = self.reserveObject('charProc')
            self.outputStream(charprocObject, stream, extra=charprocDict)
            if charname in multi_byte_chars:
                name = self._get_xobject_glyph_name(filename, charname)
                self.multi_byte_charprocs[name] = charprocObject
            else:
                charprocs[charname] = charprocObject
        self.writeObject(fontdictObject, fontdict)
        self.writeObject(fontdescObject, descriptor)
        self.writeObject(widthsObject, widths)
        self.writeObject(charprocsObject, charprocs)
        return fontdictObject

    def embedTTFType42(font, characters, descriptor):
        """The Type 42-specific part of embedding a Truetype font"""
        fontdescObject = self.reserveObject('font descriptor')
        cidFontDictObject = self.reserveObject('CID font dictionary')
        type0FontDictObject = self.reserveObject('Type 0 font dictionary')
        cidToGidMapObject = self.reserveObject('CIDToGIDMap stream')
        fontfileObject = self.reserveObject('font file stream')
        wObject = self.reserveObject('Type 0 widths')
        toUnicodeMapObject = self.reserveObject('ToUnicode map')
        subset_str = ''.join((chr(c) for c in characters))
        _log.debug('SUBSET %s characters: %s', filename, subset_str)
        fontdata = _backend_pdf_ps.get_glyphs_subset(filename, subset_str)
        _log.debug('SUBSET %s %d -> %d', filename, os.stat(filename).st_size, fontdata.getbuffer().nbytes)
        full_font = font
        font = FT2Font(fontdata)
        cidFontDict = {'Type': Name('Font'), 'Subtype': Name('CIDFontType2'), 'BaseFont': ps_name, 'CIDSystemInfo': {'Registry': 'Adobe', 'Ordering': 'Identity', 'Supplement': 0}, 'FontDescriptor': fontdescObject, 'W': wObject, 'CIDToGIDMap': cidToGidMapObject}
        type0FontDict = {'Type': Name('Font'), 'Subtype': Name('Type0'), 'BaseFont': ps_name, 'Encoding': Name('Identity-H'), 'DescendantFonts': [cidFontDictObject], 'ToUnicode': toUnicodeMapObject}
        descriptor['FontFile2'] = fontfileObject
        self.outputStream(fontfileObject, fontdata.getvalue(), extra={'Length1': fontdata.getbuffer().nbytes})
        cid_to_gid_map = ['\x00'] * 65536
        widths = []
        max_ccode = 0
        for c in characters:
            ccode = c
            gind = font.get_char_index(ccode)
            glyph = font.load_char(ccode, flags=LOAD_NO_SCALE | LOAD_NO_HINTING)
            widths.append((ccode, cvt(glyph.horiAdvance)))
            if ccode < 65536:
                cid_to_gid_map[ccode] = chr(gind)
            max_ccode = max(ccode, max_ccode)
        widths.sort()
        cid_to_gid_map = cid_to_gid_map[:max_ccode + 1]
        last_ccode = -2
        w = []
        max_width = 0
        unicode_groups = []
        for ccode, width in widths:
            if ccode != last_ccode + 1:
                w.append(ccode)
                w.append([width])
                unicode_groups.append([ccode, ccode])
            else:
                w[-1].append(width)
                unicode_groups[-1][1] = ccode
            max_width = max(max_width, width)
            last_ccode = ccode
        unicode_bfrange = []
        for start, end in unicode_groups:
            if start > 65535:
                continue
            end = min(65535, end)
            unicode_bfrange.append(b'<%04x> <%04x> [%s]' % (start, end, b' '.join((b'<%04x>' % x for x in range(start, end + 1)))))
        unicode_cmap = self._identityToUnicodeCMap % (len(unicode_groups), b'\n'.join(unicode_bfrange))
        glyph_ids = []
        for ccode in characters:
            if not _font_supports_glyph(fonttype, ccode):
                gind = full_font.get_char_index(ccode)
                glyph_ids.append(gind)
        bbox = [cvt(x, nearest=False) for x in full_font.bbox]
        rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
        for charname in sorted(rawcharprocs):
            stream = rawcharprocs[charname]
            charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
            stream = stream[stream.find(b'd1') + 2:]
            charprocObject = self.reserveObject('charProc')
            self.outputStream(charprocObject, stream, extra=charprocDict)
            name = self._get_xobject_glyph_name(filename, charname)
            self.multi_byte_charprocs[name] = charprocObject
        cid_to_gid_map = ''.join(cid_to_gid_map).encode('utf-16be')
        self.outputStream(cidToGidMapObject, cid_to_gid_map)
        self.outputStream(toUnicodeMapObject, unicode_cmap)
        descriptor['MaxWidth'] = max_width
        self.writeObject(cidFontDictObject, cidFontDict)
        self.writeObject(type0FontDictObject, type0FontDict)
        self.writeObject(fontdescObject, descriptor)
        self.writeObject(wObject, w)
        return type0FontDictObject
    ps_name = self._get_subsetted_psname(font.postscript_name, font.get_charmap())
    ps_name = ps_name.encode('ascii', 'replace')
    ps_name = Name(ps_name)
    pclt = font.get_sfnt_table('pclt') or {'capHeight': 0, 'xHeight': 0}
    post = font.get_sfnt_table('post') or {'italicAngle': (0, 0)}
    ff = font.face_flags
    sf = font.style_flags
    flags = 0
    symbolic = False
    if ff & FIXED_WIDTH:
        flags |= 1 << 0
    if 0:
        flags |= 1 << 1
    if symbolic:
        flags |= 1 << 2
    else:
        flags |= 1 << 5
    if sf & ITALIC:
        flags |= 1 << 6
    if 0:
        flags |= 1 << 16
    if 0:
        flags |= 1 << 17
    if 0:
        flags |= 1 << 18
    descriptor = {'Type': Name('FontDescriptor'), 'FontName': ps_name, 'Flags': flags, 'FontBBox': [cvt(x, nearest=False) for x in font.bbox], 'Ascent': cvt(font.ascender, nearest=False), 'Descent': cvt(font.descender, nearest=False), 'CapHeight': cvt(pclt['capHeight'], nearest=False), 'XHeight': cvt(pclt['xHeight']), 'ItalicAngle': post['italicAngle'][1], 'StemV': 0}
    if fonttype == 3:
        return embedTTFType3(font, characters, descriptor)
    elif fonttype == 42:
        return embedTTFType42(font, characters, descriptor)