import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def drawBasics(canvas):
    saver = StateSaver(canvas)
    canvas.defaultLineColor = Color(0.7, 0.7, 1.0)
    canvas.drawLines(map(lambda i: (i * 10, 0, i * 10, 300), range(30)))
    canvas.drawLines(map(lambda i: (0, i * 10, 300, i * 10), range(30)))
    canvas.defaultLineColor = black
    canvas.drawLine(10, 200, 20, 190, color=red)
    canvas.drawEllipse(130, 30, 200, 100, fillColor=yellow, edgeWidth=4)
    canvas.drawArc(130, 30, 200, 100, 45, 50, fillColor=blue, edgeColor=navy, edgeWidth=4)
    canvas.defaultLineWidth = 4
    canvas.drawRoundRect(30, 30, 100, 100, fillColor=blue, edgeColor=maroon)
    canvas.drawCurve(20, 20, 100, 50, 50, 100, 160, 160)
    polypoints = [(160, 120), (130, 190), (210, 145), (110, 145), (190, 190)]
    canvas.drawPolygon(polypoints, fillColor=lime, edgeColor=red, edgeWidth=3, closed=1)
    canvas.drawRect(200, 200, 260, 260, edgeColor=yellow, edgeWidth=5)
    canvas.drawLine(200, 260, 260, 260, color=green, width=5)
    canvas.drawLine(260, 200, 260, 260, color=red, width=5)
    return canvas