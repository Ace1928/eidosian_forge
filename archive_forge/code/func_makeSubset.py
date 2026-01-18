from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def makeSubset(self, subset):
    """Create a subset of a TrueType font"""
    output = TTFontMaker()
    glyphMap = [0]
    glyphSet = {0: 0}
    codeToGlyph = {}
    for code in subset:
        if code in self.charToGlyph:
            originalGlyphIdx = self.charToGlyph[code]
        else:
            originalGlyphIdx = 0
        if originalGlyphIdx not in glyphSet:
            glyphSet[originalGlyphIdx] = len(glyphMap)
            glyphMap.append(originalGlyphIdx)
        codeToGlyph[code] = glyphSet[originalGlyphIdx]
    start = self.get_table_pos('glyf')[0]
    n = 0
    while n < len(glyphMap):
        originalGlyphIdx = glyphMap[n]
        glyphPos = self.glyphPos[originalGlyphIdx]
        glyphLen = self.glyphPos[originalGlyphIdx + 1] - glyphPos
        n += 1
        if not glyphLen:
            continue
        self.seek(start + glyphPos)
        numberOfContours = self.read_short()
        if numberOfContours < 0:
            self.skip(8)
            flags = GF_MORE_COMPONENTS
            while flags & GF_MORE_COMPONENTS:
                flags = self.read_ushort()
                glyphIdx = self.read_ushort()
                if glyphIdx not in glyphSet:
                    glyphSet[glyphIdx] = len(glyphMap)
                    glyphMap.append(glyphIdx)
                if flags & GF_ARG_1_AND_2_ARE_WORDS:
                    self.skip(4)
                else:
                    self.skip(2)
                if flags & GF_WE_HAVE_A_SCALE:
                    self.skip(2)
                elif flags & GF_WE_HAVE_AN_X_AND_Y_SCALE:
                    self.skip(4)
                elif flags & GF_WE_HAVE_A_TWO_BY_TWO:
                    self.skip(8)
    for tag in ('name', 'OS/2', 'cvt ', 'fpgm', 'prep'):
        try:
            output.add(tag, self.get_table(tag))
        except KeyError:
            pass
    post = b'\x00\x03\x00\x00' + self.get_table('post')[4:16] + b'\x00' * 16
    output.add('post', post)
    numGlyphs = len(glyphMap)
    hmtx = []
    for n in range(numGlyphs):
        aw, lsb = self.hmetrics[glyphMap[n]]
        hmtx.append(int(aw))
        hmtx.append(int(lsb))
    n = len(hmtx) - 2
    while n and hmtx[n] == hmtx[n - 2]:
        n -= 2
    n += 2
    numberOfHMetrics = n >> 1
    hmtx = hmtx[:n] + hmtx[n + 1::2]
    hmtx = pack(*['>%dH' % len(hmtx)] + hmtx)
    output.add('hmtx', hmtx)
    hhea = self.get_table('hhea')
    hhea = _set_ushort(hhea, 34, numberOfHMetrics)
    output.add('hhea', hhea)
    maxp = self.get_table('maxp')
    maxp = _set_ushort(maxp, 4, numGlyphs)
    output.add('maxp', maxp)
    entryCount = len(subset)
    length = 10 + entryCount * 2
    cmap = [0, 1, 1, 0, 0, 12, 6, length, 0, 0, entryCount] + list(map(codeToGlyph.get, subset))
    cmap = pack(*['>%dH' % len(cmap)] + cmap)
    output.add('cmap', cmap)
    glyphData = self.get_table('glyf')
    offsets = []
    glyf = []
    pos = 0
    for n in range(numGlyphs):
        offsets.append(pos)
        originalGlyphIdx = glyphMap[n]
        glyphPos = self.glyphPos[originalGlyphIdx]
        glyphLen = self.glyphPos[originalGlyphIdx + 1] - glyphPos
        data = glyphData[glyphPos:glyphPos + glyphLen]
        if glyphLen > 2 and unpack('>h', data[:2])[0] < 0:
            pos_in_glyph = 10
            flags = GF_MORE_COMPONENTS
            while flags & GF_MORE_COMPONENTS:
                flags = unpack('>H', data[pos_in_glyph:pos_in_glyph + 2])[0]
                glyphIdx = unpack('>H', data[pos_in_glyph + 2:pos_in_glyph + 4])[0]
                data = _set_ushort(data, pos_in_glyph + 2, glyphSet[glyphIdx])
                pos_in_glyph = pos_in_glyph + 4
                if flags & GF_ARG_1_AND_2_ARE_WORDS:
                    pos_in_glyph = pos_in_glyph + 4
                else:
                    pos_in_glyph = pos_in_glyph + 2
                if flags & GF_WE_HAVE_A_SCALE:
                    pos_in_glyph = pos_in_glyph + 2
                elif flags & GF_WE_HAVE_AN_X_AND_Y_SCALE:
                    pos_in_glyph = pos_in_glyph + 4
                elif flags & GF_WE_HAVE_A_TWO_BY_TWO:
                    pos_in_glyph = pos_in_glyph + 8
        glyf.append(data)
        pos = pos + glyphLen
        if pos % 4 != 0:
            padding = 4 - pos % 4
            glyf.append(b'\x00' * padding)
            pos = pos + padding
    offsets.append(pos)
    output.add('glyf', b''.join(glyf))
    loca = []
    if pos + 1 >> 1 > 65535:
        indexToLocFormat = 1
        for offset in offsets:
            loca.append(offset)
        loca = pack(*['>%dL' % len(loca)] + loca)
    else:
        indexToLocFormat = 0
        for offset in offsets:
            loca.append(offset >> 1)
        loca = pack(*['>%dH' % len(loca)] + loca)
    output.add('loca', loca)
    head = self.get_table('head')
    head = _set_ushort(head, 50, indexToLocFormat)
    output.add('head', head)
    return output.makeStream()