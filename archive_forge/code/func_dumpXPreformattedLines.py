from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def dumpXPreformattedLines(P):
    print('\n############dumpXPreforemattedLines(%s)' % str(P))
    lines = P.blPara.lines
    n = len(lines)
    outw = sys.stdout.write
    for l in range(n):
        line = lines[l]
        words = line.words
        nwords = len(words)
        outw('line%d: %d(%d)\n  ' % (l, nwords, line.wordCount))
        for w in range(nwords):
            outw(" %d:'%s'" % (w, words[w].text))
        print()