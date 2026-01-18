import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
def stringformatTest():
    canvas = PDFCanvas('bigtest1.pdf')
    x = 10
    y = canvas.defaultFont.size * 1.5
    x, y = allTagCombos(canvas, x, y)
    x, y = allTagCombos(canvas, x, y + 30, Font(face='serif'))
    x, y = allTagCombos(canvas, x, y + 30, Font(face='monospaced'))
    x, y = allTagCombos(canvas, x, y + 30, Font(face='serif'), angle=-30)
    x, y = allTagCombos(canvas, x, y + 30, Font(size=16))
    x, y = allTagCombos(canvas, x, y + 30, Font(size=9))
    x, y = allTagCombos(canvas, x, y + 30, Font(underline=1))
    x, y = allTagCombos(canvas, x, y + 30, color=red)
    sfwidth = stringWidth(canvas, '<b><sub>bold+sub</sub></b> hello <u><super>underline+super</super></u>')
    print('sw("<b><sub>bold+sub</sub></b>") = ', stringWidth(canvas, '<b><sub>bold+sub</sub></b>'))
    print('sw(" hello ") = ', stringWidth(canvas, ' hello '))
    print('sw("<u><super>underline+super</super></u>") = ', stringWidth(canvas, '<u><super>underline+super</super></u>'))
    pwidth1 = canvas.stringWidth('bold+sub', Font(size=canvas.defaultFont.size - sizedelta, bold=1))
    print('pwidth1 = ', pwidth1)
    pwidth2 = canvas.stringWidth(' hello ')
    print('pwidth2 = ', pwidth2)
    pwidth3 = canvas.stringWidth('underline+super', Font(size=canvas.defaultFont.size - sizedelta, underline=1))
    print('pwidth3 = ', pwidth3)
    print('sfwidth = ', sfwidth, ' pwidth = ', pwidth1 + pwidth2 + pwidth3)
    canvas = PDFCanvas('bigtest2.pdf')
    x = 10
    y = canvas.defaultFont.size * 1.5
    drawString(canvas, '&alpha; &beta; <chi/> &Delta; <delta/>', x, y, Font(size=16), color=blue)
    print('line starting with alpha should be font size 16')
    y = y + 30
    drawString(canvas, '&epsiv; &eta; &Gamma; <gamma/>', x, y, color=green)
    y = y + 30
    drawString(canvas, '&iota; &kappa; &Lambda; <lambda/>', x, y, color=blue)
    y = y + 30
    drawString(canvas, '<u>&mu;</u> &nu; <b>&Omega;</b> <omega/>', x, y, color=green)
    print('mu should be underlined, Omega should be big and bold')
    y = y + 30
    drawString(canvas, '&omicron; &Phi; &phi; <phiv/>', x, y, color=blue)
    y = y + 30
    drawString(canvas, '&Pi; &pi; &piv; <Psi/> &psi; &rho;', x, y, color=green)
    y = y + 30
    drawString(canvas, '<u>&Sigma; &sigma; &sigmav; <tau/></u>', x, y, color=blue)
    print('line starting with sigma should be completely underlined')
    y = y + 30
    drawString(canvas, '&Theta; &theta; &thetav; <Xi/> &xi; &zeta;', x, y, color=green)
    y = y + 30
    drawString(canvas, "That's &alpha;ll <u>folks</u><super>&omega;</super>", x, y)
    canvas.flush()