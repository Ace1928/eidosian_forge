import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def drawStrings(canvas):
    """Checks font metrics, and also illustrates the standard fonts."""
    saver = StateSaver(canvas)

    def Write(canvas, s, font, curs):
        if font:
            canvas.defaultFont = font
        text = s
        while text and text[-1] == '\n':
            text = text[:-1]
        canvas.drawString(text, x=curs[0], y=curs[1])
        if s[-1] == '\n':
            curs[0] = 10
            curs[1] = curs[1] + canvas.fontHeight() + canvas.fontDescent()
        else:
            curs[0] = curs[0] + canvas.stringWidth(s)

    def StandardFonts(canvas, Write):
        canvas.defaultLineColor = black
        curs = [10, 70]
        for size in (12, 18):
            for fontname in ('times', 'courier', 'helvetica', 'symbol', 'monospaced', 'serif', 'sansserif'):
                curs[0] = 10
                curs[1] = curs[1] + size * 1.5
                Write(canvas, '%s %d ' % (fontname, size), Font(face=fontname, size=size), curs)
                Write(canvas, 'bold ', Font(face=fontname, size=size, bold=1), curs)
                Write(canvas, 'italic ', Font(face=fontname, size=size, italic=1), curs)
                Write(canvas, 'underline', Font(face=fontname, size=size, underline=1), curs)
    CenterAndBox(canvas, 'spam, spam, spam, baked beans, and spam!')
    StandardFonts(canvas, Write)
    return canvas