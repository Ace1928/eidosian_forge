from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def extractInfo(self, charInfo=1):
    """
        Extract typographic information from the loaded font file.

        The following attributes will be set::
        
            name         PostScript font name
            flags        Font flags
            ascent       Typographic ascender in 1/1000ths of a point
            descent      Typographic descender in 1/1000ths of a point
            capHeight    Cap height in 1/1000ths of a point (0 if not available)
            bbox         Glyph bounding box [l,t,r,b] in 1/1000ths of a point
            _bbox        Glyph bounding box [l,t,r,b] in unitsPerEm
            unitsPerEm   Glyph units per em
            italicAngle  Italic angle in degrees ccw
            stemV        stem weight in 1/1000ths of a point (approximate)
        
        If charInfo is true, the following will also be set::
        
            defaultWidth   default glyph width in 1/1000ths of a point
            charWidths     dictionary of character widths for every supported UCS character
                           code
        
        This will only work if the font has a Unicode cmap (platform 3,
        encoding 1, format 4 or platform 0 any encoding format 4).  Setting
        charInfo to false avoids this requirement
        
        """
    name_offset = self.seek_table('name')
    format = self.read_ushort()
    if format != 0:
        raise TTFError('Unknown name table format (%d)' % format)
    numRecords = self.read_ushort()
    string_data_offset = name_offset + self.read_ushort()
    names = {1: None, 2: None, 3: None, 4: None, 6: None}
    K = list(names.keys())
    nameCount = len(names)
    for i in range(numRecords):
        platformId = self.read_ushort()
        encodingId = self.read_ushort()
        languageId = self.read_ushort()
        nameId = self.read_ushort()
        length = self.read_ushort()
        offset = self.read_ushort()
        if nameId not in K:
            continue
        N = None
        if platformId == 3 and encodingId == 1 and (languageId == 1033):
            opos = self._pos
            try:
                self.seek(string_data_offset + offset)
                if length % 2 != 0:
                    raise TTFError('PostScript name is UTF-16BE string of odd length')
                N = TTFNameBytes(self.get_chunk(string_data_offset + offset, length), 'utf_16_be')
            finally:
                self._pos = opos
        elif platformId == 1 and encodingId == 0 and (languageId == 0):
            N = TTFNameBytes(self.get_chunk(string_data_offset + offset, length), 'mac_roman')
        if N and names[nameId] == None:
            names[nameId] = N
            nameCount -= 1
            if nameCount == 0:
                break
    if names[6] is not None:
        psName = names[6]
    elif names[4] is not None:
        psName = names[4]
    elif names[1] is not None:
        psName = names[1]
    else:
        psName = None
    if not psName:
        if rl_config.autoGenerateTTFMissingTTFName:
            fn = self.filename
            if fn:
                bfn = os.path.splitext(os.path.basename(fn))[0]
            if not fn:
                psName = bytestr('_RL_%s_%s_TTF' % (time.time(), self.__class__._agfnc))
                self.__class__._agfnc += 1
            else:
                psName = self._agfnm.get(fn, '')
                if not psName:
                    if bfn:
                        psName = bytestr('_RL_%s_TTF' % bfn)
                    else:
                        psName = bytestr('_RL_%s_%s_TTF' % (time.time(), self.__class__._agfnc))
                        self.__class__._agfnc += 1
                    self._agfnm[fn] = psName
        else:
            raise TTFError('Could not find PostScript font name')
    psName = psName.__class__(psName.replace(b' ', b'-'))
    for c in psName:
        if char2int(c) > 126 or c in b' [](){}<>/%':
            raise TTFError('psName=%r contains invalid character %s' % (psName, ascii(c)))
    self.name = psName
    self.familyName = names[1] or psName
    self.styleName = names[2] or 'Regular'
    self.fullName = names[4] or psName
    self.uniqueFontID = names[3] or psName
    try:
        self.seek_table('head')
    except:
        raise TTFError('head table not found ttf name=%s' % self.name)
    ver_maj, ver_min = (self.read_ushort(), self.read_ushort())
    if ver_maj != 1:
        raise TTFError('Unknown head table version %d.%04x' % (ver_maj, ver_min))
    self.fontRevision = (self.read_ushort(), self.read_ushort())
    self.skip(4)
    magic = self.read_ulong()
    if magic != 1594834165:
        raise TTFError('Invalid head table magic %04x' % magic)
    self.skip(2)
    self.unitsPerEm = unitsPerEm = self.read_ushort()
    scale = lambda x, unitsPerEm=unitsPerEm: x * 1000.0 / unitsPerEm
    self.skip(16)
    xMin = self.read_short()
    yMin = self.read_short()
    xMax = self.read_short()
    yMax = self.read_short()
    self.bbox = list(map(scale, [xMin, yMin, xMax, yMax]))
    self.skip(3 * 2)
    indexToLocFormat = self.read_ushort()
    glyphDataFormat = self.read_ushort()
    subsettingAllowed = True
    if 'OS/2' in self.table:
        self.seek_table('OS/2')
        version = self.read_ushort()
        self.skip(2)
        usWeightClass = self.read_ushort()
        self.skip(2)
        fsType = self.read_ushort()
        if fsType == 2 or fsType & 768:
            subsettingAllowed = os.path.basename(self.filename) not in rl_config.allowTTFSubsetting
        self.skip(58)
        sTypoAscender = self.read_short()
        sTypoDescender = self.read_short()
        self.ascent = scale(sTypoAscender)
        self.descent = scale(sTypoDescender)
        if version > 1:
            self.skip(16)
            sCapHeight = self.read_short()
            self.capHeight = scale(sCapHeight)
        else:
            self.capHeight = self.ascent
    else:
        usWeightClass = 500
        self.ascent = scale(yMax)
        self.descent = scale(yMin)
        self.capHeight = self.ascent
    self.stemV = 50 + int((usWeightClass / 65.0) ** 2)
    self.seek_table('post')
    ver_maj, ver_min = (self.read_ushort(), self.read_ushort())
    if ver_maj not in (1, 2, 3, 4):
        raise TTFError('Unknown post table version %d.%04x' % (ver_maj, ver_min))
    self.italicAngle = self.read_short() + self.read_ushort() / 65536.0
    self.underlinePosition = self.read_short()
    self.underlineThickness = self.read_short()
    isFixedPitch = self.read_ulong()
    self.flags = FF_SYMBOLIC
    if self.italicAngle != 0:
        self.flags = self.flags | FF_ITALIC
    if usWeightClass >= 600:
        self.flags = self.flags | FF_FORCEBOLD
    if isFixedPitch:
        self.flags = self.flags | FF_FIXED
    self.seek_table('hhea')
    ver_maj, ver_min = (self.read_ushort(), self.read_ushort())
    if ver_maj != 1:
        raise TTFError('Unknown hhea table version %d.%04x' % (ver_maj, ver_min))
    self.skip(28)
    metricDataFormat = self.read_ushort()
    if metricDataFormat != 0:
        raise TTFError('Unknown horizontal metric data format (%d)' % metricDataFormat)
    numberOfHMetrics = self.read_ushort()
    if numberOfHMetrics == 0:
        raise TTFError('Number of horizontal metrics is 0')
    self.seek_table('maxp')
    ver_maj, ver_min = (self.read_ushort(), self.read_ushort())
    if ver_maj != 1:
        raise TTFError('Unknown maxp table version %d.%04x' % (ver_maj, ver_min))
    self.numGlyphs = numGlyphs = self.read_ushort()
    if not subsettingAllowed:
        if self.numGlyphs > 255:
            raise TTFError('Font does not allow subsetting/embedding (%04X)' % fsType)
        else:
            self._full_font = True
    else:
        self._full_font = False
    if not charInfo:
        self.charToGlyph = None
        self.defaultWidth = None
        self.charWidths = None
        return
    if glyphDataFormat != 0:
        raise TTFError('Unknown glyph data format (%d)' % glyphDataFormat)
    cmap_offset = self.seek_table('cmap')
    cmapVersion = self.read_ushort()
    cmapTableCount = self.read_ushort()
    if cmapTableCount == 0 and cmapVersion != 0:
        cmapTableCount, cmapVersion = (cmapVersion, cmapTableCount)
    encoffs = None
    enc = 0
    for n in range(cmapTableCount):
        platform = self.read_ushort()
        encoding = self.read_ushort()
        offset = self.read_ulong()
        if platform == 3:
            enc = 1
            encoffs = offset
        elif platform == 1 and encoding == 0 and (enc != 1):
            enc = 2
            encoffs = offset
        elif platform == 1 and encoding == 1:
            enc = 1
            encoffs = offset
        elif platform == 0 and encoding != 5:
            enc = 1
            encoffs = offset
    if encoffs is None:
        raise TTFError('could not find a suitable cmap encoding')
    encoffs += cmap_offset
    self.seek(encoffs)
    fmt = self.read_ushort()
    self.charToGlyph = charToGlyph = {}
    glyphToChar = {}
    if fmt in (13, 12, 10, 8):
        self.skip(2)
        length = self.read_ulong()
        lang = self.read_ulong()
    else:
        length = self.read_ushort()
        lang = self.read_ushort()
    if fmt == 0:
        T = [self.read_uint8() for i in range(length - 6)]
        for unichar in range(min(256, self.numGlyphs, len(T))):
            glyph = T[unichar]
            charToGlyph[unichar] = glyph
            glyphToChar.setdefault(glyph, []).append(unichar)
    elif fmt == 4:
        limit = encoffs + length
        segCount = int(self.read_ushort() / 2.0)
        self.skip(6)
        endCount = [self.read_ushort() for _ in range(segCount)]
        self.skip(2)
        startCount = [self.read_ushort() for _ in range(segCount)]
        idDelta = [self.read_short() for _ in range(segCount)]
        idRangeOffset_start = self._pos
        idRangeOffset = [self.read_ushort() for _ in range(segCount)]
        for n in range(segCount):
            for unichar in range(startCount[n], endCount[n] + 1):
                if idRangeOffset[n] == 0:
                    glyph = unichar + idDelta[n] & 65535
                else:
                    offset = (unichar - startCount[n]) * 2 + idRangeOffset[n]
                    offset = idRangeOffset_start + 2 * n + offset
                    if offset >= limit:
                        glyph = 0
                    else:
                        glyph = self.get_ushort(offset)
                        if glyph != 0:
                            glyph = glyph + idDelta[n] & 65535
                charToGlyph[unichar] = glyph
                glyphToChar.setdefault(glyph, []).append(unichar)
    elif fmt == 6:
        first = self.read_ushort()
        count = self.read_ushort()
        for glyph in range(first, first + count):
            unichar = self.read_ushort()
            charToGlyph[unichar] = glyph
            glyphToChar.setdefault(glyph, []).append(unichar)
    elif fmt == 10:
        first = self.read_ulong()
        count = self.read_ulong()
        for glyph in range(first, first + count):
            unichar = self.read_ushort()
            charToGlyph[unichar] = glyph
            glyphToChar.setdefault(glyph, []).append(unichar)
    elif fmt == 12:
        segCount = self.read_ulong()
        for n in range(segCount):
            start = self.read_ulong()
            end = self.read_ulong()
            inc = self.read_ulong() - start
            for unichar in range(start, end + 1):
                glyph = unichar + inc
                charToGlyph[unichar] = glyph
                glyphToChar.setdefault(glyph, []).append(unichar)
    elif fmt == 13:
        segCount = self.read_ulong()
        for n in range(segCount):
            start = self.read_ulong()
            end = self.read_ulong()
            gid = self.read_ulong()
            for unichar in range(start, end + 1):
                charToGlyph[unichar] = gid
                glyphToChar.setdefault(gid, []).append(unichar)
    elif fmt == 2:
        T = [self.read_ushort() for i in range(256)]
        maxSHK = max(T)
        SH = []
        for i in range(maxSHK + 1):
            firstCode = self.read_ushort()
            entryCount = self.read_ushort()
            idDelta = self.read_ushort()
            idRangeOffset = self.read_ushort() - (maxSHK - i) * 8 - 2 >> 1
            SH.append(CMapFmt2SubHeader(firstCode, entryCount, idDelta, idRangeOffset))
        entryCount = length - (self._pos - (cmap_offset + encoffs)) >> 1
        glyphs = [self.read_short() for i in range(entryCount)]
        last = -1
        for unichar in range(256):
            if T[unichar] == 0:
                if last != -1:
                    glyph = 0
                elif unichar < SH[0].firstCode or unichar >= SH[0].firstCode + SH[0].entryCount or SH[0].idRangeOffset + (unichar - SH[0].firstCode) >= entryCount:
                    glyph = 0
                else:
                    glyph = glyphs[SH[0].idRangeOffset + (unichar - SH[0].firstCode)]
                    if glyph != 0:
                        glyph += SH[0].idDelta
                if glyph != 0 and glyph < self.numGlyphs:
                    charToGlyph[unichar] = glyph
                    glyphToChar.setdefault(glyph, []).append(unichar)
            else:
                k = T[unichar]
                for j in range(SH[k].entryCount):
                    if SH[k].idRangeOffset + j >= entryCount:
                        glyph = 0
                    else:
                        glyph = glyphs[SH[k].idRangeOffset + j]
                        if glyph != 0:
                            glyph += SH[k].idDelta
                    if glyph != 0 and glyph < self.numGlyphs:
                        enc = unichar << 8 | j + SH[k].firstCode
                        charToGlyph[enc] = glyph
                        glyphToChar.setdefault(glyph, []).append(enc)
                if last == -1:
                    last = unichar
    else:
        raise ValueError('Unsupported cmap encoding format %d' % fmt)
    self.seek_table('hmtx')
    aw = None
    self.charWidths = charWidths = {}
    self.hmetrics = []
    for glyph in range(numberOfHMetrics):
        aw, lsb = (self.read_ushort(), self.read_ushort())
        self.hmetrics.append((aw, lsb))
        aw = scale(aw)
        if glyph == 0:
            self.defaultWidth = aw
        if glyph in glyphToChar:
            for char in glyphToChar[glyph]:
                charWidths[char] = aw
    for glyph in range(numberOfHMetrics, numGlyphs):
        lsb = self.read_ushort()
        self.hmetrics.append((aw, lsb))
        if glyph in glyphToChar:
            for char in glyphToChar[glyph]:
                charWidths[char] = aw
    if 'loca' not in self.table:
        raise TTFError('missing location table')
    self.seek_table('loca')
    self.glyphPos = []
    if indexToLocFormat == 0:
        for n in range(numGlyphs + 1):
            self.glyphPos.append(self.read_ushort() << 1)
    elif indexToLocFormat == 1:
        for n in range(numGlyphs + 1):
            self.glyphPos.append(self.read_ulong())
    else:
        raise TTFError('Unknown location table format (%d)' % indexToLocFormat)
    if 32 in charToGlyph:
        charToGlyph[160] = charToGlyph[32]
        charWidths[160] = charWidths[32]
    elif 160 in charToGlyph:
        charToGlyph[32] = charToGlyph[160]
        charWidths[32] = charWidths[160]