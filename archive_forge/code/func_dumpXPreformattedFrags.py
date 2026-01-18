from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def dumpXPreformattedFrags(P):
    print('\n############dumpXPreforemattedFrags(%s)' % str(P))
    frags = P.frags
    n = len(frags)
    for l in range(n):
        print("frag%d: '%s'" % (l, frags[l].text))
    outw = sys.stdout.write
    l = 0
    for L in _getFragLines(frags):
        n = 0
        for W in _getFragWords(L, 360):
            outw('frag%d.%d: size=%d' % (l, n, W[0]))
            n = n + 1
            for w in W[1:]:
                outw(" '%s'" % w[1])
            print()
        l = l + 1