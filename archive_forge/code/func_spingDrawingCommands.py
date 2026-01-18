from sping import colors
from sping.TK import TKCanvas
from Tkinter import *
def spingDrawingCommands(self, anySpingCanvas):
    anySpingCanvas.drawRect(10, 10, 100, 100, edgeColor=colors.blue, fillColor=colors.green)
    anySpingCanvas.drawRect(400, 400, 500, 500, edgeColor=colors.blue, fillColor=colors.lightblue)
    anySpingCanvas.drawRect(30, 400, 130, 500, edgeColor=colors.blue, fillColor=colors.yellow)