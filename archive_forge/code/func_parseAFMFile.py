import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def parseAFMFile(afmFileName):
    """Quick and dirty - gives back a top-level dictionary
    with top-level items, and a 'widths' key containing
    a dictionary of glyph names and widths.  Just enough
    needed for embedding.  A better parser would accept
    options for what data you wwanted, and preserve the
    order."""
    lines = open_and_readlines(afmFileName, 'r')
    if len(lines) <= 1:
        if lines:
            lines = lines[0].split('\r')
        if len(lines) <= 1:
            raise ValueError("AFM file %s hasn't enough data" % afmFileName)
    topLevel = {}
    glyphLevel = []
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if not l.lower().startswith('comment')]
    inMetrics = 0
    for line in lines:
        if line[0:16] == 'StartCharMetrics':
            inMetrics = 1
        elif line[0:14] == 'EndCharMetrics':
            inMetrics = 0
        elif inMetrics:
            chunks = line.split(';')
            chunks = [chunk.strip() for chunk in chunks]
            cidChunk, widthChunk, nameChunk = chunks[0:3]
            l, r = cidChunk.split()
            assert l == 'C', 'bad line in font file %s' % line
            cid = int(r)
            l, r = widthChunk.split()
            assert l == 'WX', 'bad line in font file %s' % line
            try:
                width = int(r)
            except ValueError:
                width = float(r)
            l, r = nameChunk.split()
            assert l == 'N', 'bad line in font file %s' % line
            name = r
            glyphLevel.append((cid, width, name))
    inHeader = 0
    for line in lines:
        if line[0:16] == 'StartFontMetrics':
            inHeader = 1
        if line[0:16] == 'StartCharMetrics':
            inHeader = 0
        elif inHeader:
            if line[0:7] == 'Comment':
                pass
            try:
                left, right = line.split(' ', 1)
            except:
                raise ValueError("Header information error in afm %s: line='%s'" % (afmFileName, line))
            try:
                right = int(right)
            except:
                pass
            topLevel[left] = right
    return (topLevel, glyphLevel)