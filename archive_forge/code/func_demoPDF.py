import formatter
import string
from types import *
import htmllib
import piddle
def demoPDF(html):
    import piddlePDF
    pc = piddlePDF.PDFCanvas((750, 1000), 'HTMLPiddler.pdf')
    pc.drawLine(100, 100, 250, 150, color=piddle.green)
    pc.drawRect(100, 100, 650, 900, edgeColor=piddle.pink)
    ptt = HTMLPiddler(html, (250, 150), (100, 650))
    ptt.renderOn(pc)
    pc.save()