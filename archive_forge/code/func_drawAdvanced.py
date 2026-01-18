import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def drawAdvanced(canvas):
    saver = StateSaver(canvas)
    figure = [(figureCurve, 20, 20, 100, 50, 50, 100, 160, 160), (figureLine, 200, 200, 250, 150), (figureArc, 50, 10, 250, 150, 10, 90)]
    canvas.drawFigure(figure, fillColor=yellow, edgeWidth=4)
    try:
        from PIL import Image
    except ImportError:
        canvas.drawString('PIL not available!', 20, 200)
        Image = None
    if Image:
        img = Image.open('python.gif')
        canvas.drawImage(img, 120, 50, 120 + 32, 50 + 64)
        canvas.drawImage(img, 0, 210, 300, 210 + 32)
    return canvas