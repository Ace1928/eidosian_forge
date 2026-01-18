import formatter
import string
from types import *
import htmllib
import piddle
def demoTK(html):
    import piddleTK
    pc = piddleTK.TKCanvas((800, 600))
    pc.drawLine(0, 0, 50, 50, color=piddle.green)
    pc.drawRect(10, 10, 590, 790, edgeColor=piddle.pink)
    ptt = HTMLPiddler(html, (50, 50), (10, 790))
    pc.flush()
    ptt.renderOn(pc)