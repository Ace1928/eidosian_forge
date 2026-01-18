import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def drawMinimal(canvas):
    saver = StateSaver(canvas)
    size = canvas.size
    canvas.defaultLineColor = green
    canvas.drawLine(1, 1, size[0] - 1, size[1] - 1)
    canvas.drawLine(1, size[1] - 1, size[0] - 1, 1)
    canvas.drawRect(1, 1, size[0] - 1, size[1] - 1, edgeWidth=5)
    return canvas